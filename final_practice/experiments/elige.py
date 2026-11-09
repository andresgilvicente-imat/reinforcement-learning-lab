"""Elige configuración (y semilla) teniendo en cuenta LAS DOS clasificaciones.

    python experiments/elige.py                       # todos los e*_*.csv de resultados/
    python experiments/elige.py --algoritmo sarsa --figura
    python experiments/elige.py resultados/e2_sarsa.csv resultados/e3_sarsa.csv

Las dos clasificaciones tiran en sentidos opuestos (tiles gruesos -> más
llegadas, tiles finos -> caminos más precisos), así que no hay "un mejor":

1. **Frente de Pareto.** Una configuración domina a otra si tiene al menos
   tantas llegadas Y como mucho tantos pasos de más, y es estrictamente mejor
   en alguna. Las no dominadas forman el frente: todo lo demás es peor en algo
   sin ser mejor en nada, así que se descarta.

2. **Regla de elección dentro del frente.** Las llegadas medias tienen ruido
   (posiciones y semillas). Se consideran *empatadas con la mejor* todas las
   configuraciones cuya media está a menos de ``tol`` llegadas de la máxima,
   con ``tol = 2 · error estándar`` de esa media. Entre las empatadas se elige
   la de MENOS pasos de más: ganar una llegada que puede ser ruido no compensa
   perder precisión que sí es real.

3. **Semilla.** Dentro de la configuración elegida se repite lo mismo entre
   sus semillas (solo en posiciones de selección; la estimación honesta es la
   del conjunto de test).

Solo se usan filas del conjunto de selección; las de test se ignoran.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTADOS = Path(__file__).resolve().parent / "resultados"
CLAVES = ("algoritmo", "rho", "n_tiles", "num_tilings", "q0", "eps", "recompensa")

DEFECTOS = {"recompensa": "base"}


def _v(r, k):
    """Valor de una columna; los CSV antiguos no tienen 'recompensa'."""
    return r.get(k) or DEFECTOS.get(k, "")


def tiles_txt(n):
    """'12' -> '12×12'; '12x16' -> '12×16'."""
    p = str(n).split("x")
    return f"{p[0]}×{p[-1]}"


def tiles_args(n):
    p = str(n).split("x")
    return f"{p[0]} {p[-1]}"


def _tk(n):
    return tuple(int(v) for v in str(n).split("x"))
N_POS = 1000
MIN_LLEGADAS = 500  # regla de la clasificación 2


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def carga(rutas):
    """Filas de selección, sin duplicados (misma configuración y semilla)."""
    filas = {}
    for ruta in rutas:
        with open(ruta, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("conjunto", "sel") != "sel":
                    continue
                filas[tuple(_v(r, k) for k in CLAVES) + (r["seed"],)] = r
    return list(filas.values())


def agrupa(filas):
    grupos = defaultdict(list)
    for r in filas:
        grupos[tuple(_v(r, k) for k in CLAVES)].append(r)
    out = []
    for clave, rs in grupos.items():
        lleg = np.array([_f(r["llegadas"]) for r in rs])
        extra = np.array([_f(r["extra"]) for r in rs])
        n = len(rs)
        p = np.clip(lleg.mean() / N_POS, 1e-6, 1 - 1e-6)
        # error estándar de la media de llegadas: ruido de posiciones
        # (binomial) + ruido entre semillas, el mayor de los dos
        se_pos = np.sqrt(p * (1 - p) * N_POS / n)
        se_sem = lleg.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        out.append(dict(
            zip(CLAVES, clave), n=n, filas=rs,
            lleg=lleg.mean(), lleg_min=lleg.min(), se=max(se_pos, se_sem),
            extra=np.nanmean(extra) if np.isfinite(extra).any() else np.inf,
            diverge=sum(int(r["diverge"]) for r in rs),
        ))
    return out


def pareto(items, k_lleg="lleg", k_extra="extra"):
    frente = []
    for a in items:
        dominado = any(
            b is not a and b[k_lleg] >= a[k_lleg] and b[k_extra] <= a[k_extra]
            and (b[k_lleg] > a[k_lleg] or b[k_extra] < a[k_extra])
            for b in items
        )
        if not dominado:
            frente.append(a)
    return sorted(frente, key=lambda r: -r[k_lleg])


def elige(frente, tol=None, k_lleg="lleg", k_extra="extra"):
    mejor = max(frente, key=lambda r: r[k_lleg])
    tol = 2 * mejor.get("se", 0) if tol is None else tol
    empatadas = [r for r in frente if r[k_lleg] >= mejor[k_lleg] - tol]
    return min(empatadas, key=lambda r: r[k_extra]), tol


def nombre(r):
    alpha = float(r["rho"]) / int(r["num_tilings"])
    return (f"ρ={r['rho']} (α={alpha:.4g}) · {tiles_txt(r['n_tiles'])} · "
            f"{r['num_tilings']} til. · q0={r['q0']} · ε={r['eps']} · recompensa={r['recompensa']}")


def informa(algoritmo, configs, tol_usuario):
    validas = [c for c in configs if c["lleg_min"] >= MIN_LLEGADAS and not c["diverge"]]
    frente = pareto(validas)
    eleg, tol = elige(frente, tol_usuario)

    print(f"\n{'=' * 100}\n{algoritmo.upper()}: {len(configs)} configuraciones, "
          f"{len(validas)} válidas (sin divergencias y peor semilla ≥ {MIN_LLEGADAS}), "
          f"{len(frente)} en el frente de Pareto\n{'=' * 100}")
    print(f"{'':2}{'llegadas':>9} {'±2se':>5} {'peor':>5} {'extra':>6} {'sem':>4}  configuración")
    for r in frente:
        marca = "★" if r is eleg else ("≈" if r["lleg"] >= max(f["lleg"] for f in frente) - tol else " ")
        print(f"{marca:2}{r['lleg']:9.1f} {2 * r['se']:5.1f} {r['lleg_min']:5.0f} "
              f"{r['extra']:6.2f} {r['n']:4d}  {nombre(r)}")
    print(f"\n≈ = llegadas empatadas con la mejor (tolerancia {tol:.1f} llegadas); "
          f"★ = elegida: la de menos pasos de más entre las empatadas.")

    # semilla dentro de la elegida
    semillas = [dict(seed=r["seed"], lleg=_f(r["llegadas"]), extra=_f(r["extra"]))
                for r in eleg["filas"]]
    # p acotada: con 1000/1000 el error binomial saldría 0 y nada empataría
    p_max = min(max(s["lleg"] for s in semillas) / N_POS, 1 - 1 / N_POS)
    tol_sem = 2 * np.sqrt(p_max * (1 - p_max) * N_POS)
    sem_eleg, _ = elige(pareto(semillas), tol_sem)
    print(f"\nSemillas de la elegida (posiciones de selección; tolerancia {tol_sem:.1f}):")
    for s in sorted(semillas, key=lambda s: (-s["lleg"], s["extra"])):
        print(f"  {'★' if s is sem_eleg else ' '} seed={s['seed']:>3}  llegadas={s['lleg']:5.0f}  "
              f"extra={s['extra']:.2f}")
    alpha = float(eleg["rho"]) / int(eleg["num_tilings"])
    print(f"\n→ python main.py --algoritmo {algoritmo} --alpha {alpha:.6g} "
          f"--n_tiles {tiles_args(eleg['n_tiles'])} --num_tilings {eleg['num_tilings']} "
          f"--q0 {_f(eleg['q0']):g} --seed {sem_eleg['seed']} --guarda <G> --grupo <G>")
    if eleg["recompensa"] != "base":
        print("  (antes: copia experiments/recompensa_shaping.py en el hueco de src/recompensa.py)")
    if eleg["eps"] != "codigo":
        print(f"  (antes: escribe ε = {eleg['eps']} en epsilon_{algoritmo})")
    return validas, frente, eleg


def figura(resultados, salida):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(resultados), figsize=(6.5 * len(resultados), 5),
                             squeeze=False, constrained_layout=True)
    for ax, (alg, (validas, frente, eleg)) in zip(axes[0], resultados.items()):
        ax.scatter([r["extra"] for r in validas], [r["lleg"] / 10 for r in validas],
                   s=14, color="0.75", label="configuraciones válidas")
        fx = [r["extra"] for r in frente]
        fy = [r["lleg"] / 10 for r in frente]
        orden = np.argsort(fx)
        ax.plot(np.array(fx)[orden], np.array(fy)[orden], "o-", color="C0",
                label="frente de Pareto")
        ax.errorbar(fx, fy, yerr=[2 * r["se"] / 10 for r in frente], fmt="none",
                    color="C0", alpha=0.4)
        ax.scatter([eleg["extra"]], [eleg["lleg"] / 10], marker="*", s=300,
                   color="C3", zorder=5, label="elegida")
        ax.annotate(f"{tiles_txt(eleg['n_tiles'])}·{eleg['num_tilings']} til. {eleg['recompensa']}"
                    f"\nρ={eleg['rho']} ε={eleg['eps']}",
                    (eleg["extra"], eleg["lleg"] / 10), textcoords="offset points",
                    xytext=(12, -28), fontsize=8, color="C3")
        ax.set_xlabel("pasos de más (media) ← mejor")
        ax.set_ylabel("% llegadas (media de semillas) ↑ mejor")
        ax.set_title(f"{alg}: llegadas vs pasos de más")
        ax.set_ylim(max(50, min(fy) - 3), 100.2)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.savefig(salida, dpi=130)
    print(f"\nFigura: {salida}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="*", help="CSVs (por defecto resultados/e*_*.csv)")
    p.add_argument("--algoritmo", choices=["sarsa", "q_learning"])
    p.add_argument("--tol", type=float, default=None,
                   help="Tolerancia de empate en llegadas (por defecto 2·error estándar).")
    p.add_argument("--figura", action="store_true")
    a = p.parse_args()

    rutas = a.csv or sorted(RESULTADOS.glob("e*_*.csv"))
    configs = agrupa(carga(rutas))
    algs = [a.algoritmo] if a.algoritmo else sorted({c["algoritmo"] for c in configs})
    resultados = {alg: informa(alg, [c for c in configs if c["algoritmo"] == alg], a.tol)
                  for alg in algs}
    if a.figura:
        figura(resultados, RESULTADOS / "pareto.png")


if __name__ == "__main__":
    main()
