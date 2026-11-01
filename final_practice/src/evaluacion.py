"""Evaluación de la política greedy de un agente (Práctica 3 · Sesión 1).

Es la misma función con la que los profesores calculan la leaderboard: el
agente parte de cada posición de una lista, actúa con su política greedy
determinista (``AgenteLineal.accion_greedy``, sin exploración) y se anota si
llega a la zona objetivo y en cuántos pasos.

Un episodio de evaluación acaba al llegar al objetivo (éxito), al chocar
(fracaso) o al cumplir ``MAX_PASOS_EVALUACION`` pasos sin ninguna de las dos
cosas (fracaso: la política da vueltas).

Este fichero NO debe modificarse durante la práctica.
"""

from __future__ import annotations

import numpy as np

from env import NavigationEnv

# Ningún punto de partida necesita más de 60 pasos para llegar al objetivo.
MAX_PASOS_EVALUACION = 200
# Semilla de las posiciones de validación que podéis usar durante la práctica.
# Las de la leaderboard salen de otra semilla, que no se publica.
SEMILLA_VALIDACION = 2627

OBJETIVO, CHOQUE, SIN_LLEGAR = "objetivo", "choque", "sin llegar"


def _recompensa_nula(*_):
    return 0.0


def posiciones_aleatorias(n, seed=SEMILLA_VALIDACION) -> np.ndarray:
    """``n`` posiciones iniciales con la misma distribución que ``env.reset()``."""
    env = NavigationEnv(recompensa=_recompensa_nula)
    env.reset(seed=seed)
    posiciones = [env.reset()[0][:2] for _ in range(n)]
    return np.array(posiciones, dtype=float)


def evalua(agente, posiciones, max_pasos=MAX_PASOS_EVALUACION) -> dict:
    """Evalúa la política greedy de ``agente`` desde cada posición.

    Args:
        agente (AgenteLineal): agente a evaluar.
        posiciones (array-like): posiciones iniciales ``(x, y)``.
        max_pasos (int): pasos máximos por episodio.

    Returns:
        dict con:
            * ``llegadas`` (int): episodios que acaban en la zona objetivo.
            * ``tasa_exito`` (float): ``llegadas / len(posiciones)``.
            * ``pasos_medios`` (float): pasos medios de los episodios con
              éxito (``nan`` si no hay ninguno).
            * ``pasos`` (np.ndarray[int]): pasos de cada episodio.
            * ``final`` (np.ndarray[str]): ``"objetivo"``, ``"choque"`` o
              ``"sin llegar"`` en cada episodio.
    """
    env = NavigationEnv(recompensa=_recompensa_nula, max_steps=max_pasos)
    posiciones = np.asarray(posiciones, dtype=float)
    pasos = np.zeros(len(posiciones), dtype=int)
    final = np.empty(len(posiciones), dtype=object)
    for k, posicion in enumerate(posiciones):
        obs, _ = env.reset(options={"posicion": posicion})
        while True:
            obs, _, terminated, truncated, _ = env.step(agente.accion_greedy(obs))
            if terminated or truncated:
                break
        pasos[k] = env.steps
        final[k] = OBJETIVO if env.target else CHOQUE if env.collision else SIN_LLEGAR
    exito = final == OBJETIVO
    return {
        "llegadas": int(exito.sum()),
        "tasa_exito": float(exito.mean()),
        "pasos_medios": float(pasos[exito].mean()) if exito.any() else float("nan"),
        "pasos": pasos,
        "final": final.astype(str),
    }
