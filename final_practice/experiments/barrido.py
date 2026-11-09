"""Barrido de hiperparámetros sin sobreajustar a las posiciones de evaluación.

No modifica ningún fichero de ``src``: importa el código de la práctica y llama
a ``entrenamiento.entrena`` igual que ``main.py``, así que con los mismos
hiperparámetros y semilla los pesos salen idénticos a los de ``main.py``.

Protocolo
---------
* Selección: cada configuración se entrena con varias semillas y se evalúa en
  ``N_SEL`` posiciones de una semilla propia (no las 100 de validación, que son
  pocas y ruidosas). Se elige por la MEDIA entre semillas.
* Test: las posiciones de ``SEMILLA_TEST`` no se usan para elegir nada; solo
  para estimar honestamente el resultado de los agentes finales.
* Métrica 2 de la leaderboard: pasos de más sobre el camino mínimo, que se
  calcula con BFS en la retícula de 0,25 m que puede recorrer el agente.

Ejemplos (desde final_practice/):

    python experiments/barrido.py --algoritmo sarsa --rho 0.08 0.16 0.24 \
        --n_tiles 8 10 12 --num_tilings 4 8 16 --seeds 0 1 2 --salida exp/sarsa_e1.csv

``rho`` es el paso efectivo ``alpha * num_tilings`` (lo que se mueve q̂ en
cada actualización); así es comparable entre diseños con distinto nº de tilings.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agente import AgenteLineal  # noqa: E402
from entrenamiento import entrena  # noqa: E402
from env import (  # noqa: E402
    AGENT_STEP,
    MOVES,
    NavigationEnv,
    in_target,
    is_collision,
)
from evaluacion import evalua, posiciones_aleatorias  # noqa: E402
from q_learning import episodio_q_learning, epsilon_q_learning  # noqa: E402
from representacion import TileCoder  # noqa: E402
from sarsa import episodio_sarsa, epsilon_sarsa  # noqa: E402

SEMILLA_SEL = 777
SEMILLA_TEST = 4242
N_SEL = 1000
EPISODIOS = 2000
GAMMA = 0.99
W_DIVERGE = 1e4  # |w| por encima de esto: divergencia (q̂ real está en [-100, 100])

EPISODIOS_FN = {"sarsa": episodio_sarsa, "q_learning": episodio_q_learning}
EPSILON_CODIGO = {"sarsa": epsilon_sarsa, "q_learning": epsilon_q_learning}


# ------------------------------------------------------- tiles y recompensa
def tiles_xy(n_tiles):
    """'12' -> (12, 12); '12x16' -> (12, 16)."""
    partes = str(n_tiles).lower().split("x")
    nx = int(partes[0])
    return (nx, int(partes[1]) if len(partes) > 1 else nx)


def normaliza_tiles(n_tiles):
    nx, ny = tiles_xy(n_tiles)
    return str(nx) if nx == ny else f"{nx}x{ny}"


def recompensa_fn(nombre):
    """'src' -> None: el entorno usa src/recompensa.py (lo que se entrega).

    'base' y 'shaping' son copias fijas en experiments/, para que los
    experimentos antiguos se puedan reproducir aunque src/recompensa.py cambie.
    """
    if nombre == "src":
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if nombre == "shaping":
        import recompensa_shaping

        return recompensa_shaping.recompensa
    import recompensa_base

    return recompensa_base.recompensa


# ----------------------------------------------------------------- camino mínimo
def camino_minimo(x0, y0, max_desp=45):
    """Pasos mínimos desde (x0, y0) hasta la zona objetivo (BFS en la retícula)."""
    inicio = (0, 0)
    vistos = {inicio}
    cola = deque([(inicio, 0)])
    while cola:
        (i, j), d = cola.popleft()
        for dx, dy in MOVES.values():
            ni, nj = i + int(round(dx / AGENT_STEP)), j + int(round(dy / AGENT_STEP))
            if (ni, nj) in vistos or abs(ni) > max_desp or abs(nj) > max_desp:
                continue
            x, y = x0 + ni * AGENT_STEP, y0 + nj * AGENT_STEP
            if is_collision(x, y):
                continue
            if in_target(x, y):
                return d + 1
            vistos.add((ni, nj))
            cola.append(((ni, nj), d + 1))
    return np.nan


def posiciones_y_minimos(n, seed):
    pos = posiciones_aleatorias(n, seed=seed)
    return pos, np.array([camino_minimo(x, y) for x, y in pos])


# -------------------------------------------------------------- programa de eps
class ProgramaEpsilon:
    """Misma forma que el de sarsa.py: meseta, decaimiento exponencial, meseta."""

    def __init__(self, ini, fin, f1, f2):
        self.ini, self.fin, self.f1, self.f2 = ini, fin, f1, f2

    def __call__(self, k, n):
        fase_1, fase_2 = int(self.f1 * n), int(self.f2 * n)
        if k < fase_1:
            return self.ini
        if k < fase_2:
            prog = (k - fase_1) / max(1, fase_2 - fase_1 - 1)
            if self.fin <= 0:  # decaimiento lineal hasta 0
                return self.ini * (1 - prog)
            return self.ini * (self.fin / self.ini) ** prog
        return self.fin


class Divergencia(Exception):
    pass


def _vigilado(episodio):
    def f(env, agente, *args):
        res = episodio(env, agente, *args)
        if np.abs(agente.w).max() > W_DIVERGE:
            raise Divergencia
        return res

    return f


# ----------------------------------------------------------------- un trabajo
_POS = {}


def _init(pos_sel, min_sel):
    _POS["pos"], _POS["min"] = pos_sel, min_sel


def entrena_y_evalua(cfg):
    t0 = time.time()
    coder = TileCoder(n_tiles=tiles_xy(cfg["n_tiles"]), num_tilings=cfg["num_tilings"])
    env = NavigationEnv(recompensa=recompensa_fn(cfg["recompensa"]))
    agente = AgenteLineal(coder, num_acciones=4, q0=cfg["q0"])
    alpha = cfg["rho"] / cfg["num_tilings"]
    if cfg["eps"] == "codigo":
        programa = EPSILON_CODIGO[cfg["algoritmo"]]
    else:
        programa = ProgramaEpsilon(*cfg["eps"])
    fila = dict(cfg, alpha=alpha, eps=str(cfg["eps"]))
    try:
        h = entrena(
            env, agente, _vigilado(EPISODIOS_FN[cfg["algoritmo"]]), programa,
            EPISODIOS, alpha, GAMMA, seed=cfg["seed"],
        )
    except Exception:  # noqa: BLE001  divergencia (o pesos no finitos)
        return dict(fila, diverge=1, llegadas=0, tasa=0.0, choques=np.nan,
                    extra=np.nan, pasos_entreno=np.nan, segundos=time.time() - t0)
    r = evalua(agente, _POS["pos"])
    ok = r["final"] == "objetivo"
    extra = (r["pasos"][ok] - _POS["min"][ok]).mean() if ok.any() else np.nan
    return dict(
        fila, diverge=0, llegadas=r["llegadas"], tasa=r["tasa_exito"],
        choques=int((r["final"] == "choque").sum()), extra=extra,
        pasos_entreno=int(h["pasos"].sum()), wmax=float(np.abs(agente.w).max()),
        segundos=time.time() - t0,
    )


# ----------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algoritmo", choices=["sarsa", "q_learning"], required=True)
    p.add_argument("--rho", type=float, nargs="+", required=True,
                   help="Paso efectivo alpha*num_tilings.")
    p.add_argument("--n_tiles", type=str, nargs="+", default=["10"],
                   help="Tiles por dimensión: '12' (12x12) o '12x16' (x por y).")
    p.add_argument("--recompensa", choices=["src", "base", "shaping"], nargs="+",
                   default=["src"],
                   help="src = lo que haya en src/recompensa.py (lo que se entrega); "
                        "base = experiments/recompensa_base.py (+100/-100/-1); "
                        "shaping = experiments/recompensa_shaping.py")
    p.add_argument("--num_tilings", type=int, nargs="+", default=[8])
    p.add_argument("--q0", type=float, nargs="+", default=[0.0])
    p.add_argument("--eps", type=str, nargs="+", default=["codigo"],
                   help="'codigo' o 'ini,fin,f1,f2' (p. ej. 0.7,0,0.2,0.85).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_sel", type=int, default=N_SEL)
    p.add_argument("--test", action="store_true",
                   help="Evalúa en las posiciones de TEST (solo para los finales).")
    p.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--salida", type=str, required=True)
    a = p.parse_args()

    eps = [e if e == "codigo" else tuple(float(v) for v in e.split(",")) for e in a.eps]
    cfgs = [
        dict(algoritmo=a.algoritmo, rho=rho, n_tiles=normaliza_tiles(nt),
             num_tilings=ntl, q0=q0, eps=e, recompensa=rec, seed=s)
        for rho, nt, ntl, q0, e, rec, s in itertools.product(
            a.rho, a.n_tiles, a.num_tilings, a.q0, eps, a.recompensa, a.seeds)
    ]
    semilla = SEMILLA_TEST if a.test else SEMILLA_SEL
    print(f"{len(cfgs)} entrenamientos · posiciones: {a.n_sel} (semilla {semilla}) "
          f"· {a.procesos} procesos")
    pos, minimos = posiciones_y_minimos(a.n_sel, semilla)

    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    nueva = not salida.exists()
    t0 = time.time()
    with Pool(a.procesos, initializer=_init, initargs=(pos, minimos)) as pool, \
            open(salida, "a", newline="") as f:
        escritor = None
        for i, fila in enumerate(pool.imap_unordered(entrena_y_evalua, cfgs), 1):
            fila["conjunto"] = "test" if a.test else "sel"
            if escritor is None:
                campos = list(fila) + (["wmax"] if "wmax" not in fila else [])
                escritor = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
                if nueva:
                    escritor.writeheader()
            escritor.writerow(fila)
            f.flush()
            print(f"[{i}/{len(cfgs)} {time.time() - t0:5.0f}s] rho={fila['rho']} "
                  f"tiles={fila['n_tiles']} tilings={fila['num_tilings']} "
                  f"q0={fila['q0']} eps={fila['eps']} rec={fila['recompensa']} "
                  f"seed={fila['seed']} -> "
                  f"{'DIVERGE' if fila['diverge'] else ''} llegadas={fila['llegadas']} "
                  f"extra={fila['extra']:.2f}")


if __name__ == "__main__":
    main()
