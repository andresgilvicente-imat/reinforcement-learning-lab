"""Curvas de aprendizaje de los agentes finales y de los primeros campeones.

    python experiments/curvas.py     # -> resultados/curvas.png

Cada 50 episodios evalúa la política greedy en las 100 posiciones de
validación de main.py (solo para ver la evolución: no se usa para elegir).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI.parent / "src"), str(AQUI)]

from agente import AgenteLineal  # noqa: E402
from barrido import recompensa_fn  # noqa: E402
from entrenamiento import entrena  # noqa: E402
from env import NavigationEnv  # noqa: E402
from q_learning import episodio_q_learning, epsilon_q_learning  # noqa: E402
from representacion import TileCoder  # noqa: E402
from sarsa import episodio_sarsa, epsilon_sarsa  # noqa: E402

AGENTES = [
    # etiqueta, episodio, epsilon, alpha, tiles, tilings, recompensa, semilla, estilo
    ("Sarsa · final (shaping, 12×20·4)", episodio_sarsa, epsilon_sarsa, 0.04, (12, 20), 4, "shaping", 1, ("C0", "-")),
    ("Q-learning · final (shaping, 12×16·8)", episodio_q_learning, epsilon_q_learning, 0.01, (12, 16), 8, "shaping", 2, ("C1", "-")),
    ("Sarsa · recompensa base (12×12·4)", episodio_sarsa, epsilon_sarsa, 0.06, (12, 12), 4, "base", 2, ("C0", "--")),
    ("Q-learning · recompensa base (6×6·4)", episodio_q_learning, epsilon_q_learning, 0.01, (6, 6), 4, "base", 2, ("C1", "--")),
]


def main():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for nombre, ep, eps, alpha, tiles, til, rec, seed, (color, ls) in AGENTES:
        agente = AgenteLineal(TileCoder(n_tiles=tiles, num_tilings=til), 4)
        h = entrena(NavigationEnv(recompensa=recompensa_fn(rec)), agente, ep, eps,
                    2000, alpha, 0.99, seed=seed, evalua_cada=50)
        val = np.array(h["validacion"], dtype=float)
        axes[0].plot(val[:, 0], 100 * val[:, 1], ls, color=color, label=nombre)
        axes[1].plot(np.cumsum(h["pasos"]) / 1000, np.arange(1, 2001), ls, color=color,
                     label=nombre)
        print(f"{nombre}: {h['pasos'].sum()} pasos de entrenamiento, "
              f"éxito greedy final {val[-1, 1]:.0%}, "
              f"primer episodio con ≥95 %: "
              f"{int(val[np.argmax(val[:, 1] >= 0.95), 0]) if (val[:, 1] >= 0.95).any() else '-'}")
    axes[0].set_xlabel("episodio de entrenamiento")
    axes[0].set_ylabel("% llegadas de la política greedy\n(100 posiciones de validación)")
    axes[0].set_title("Aprendizaje de la política greedy")
    axes[0].set_ylim(0, 102)
    axes[1].set_xlabel("pasos de entorno acumulados (miles)")
    axes[1].set_ylabel("episodios completados")
    axes[1].set_title("Experiencia consumida")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    salida = AQUI / "resultados" / "curvas.png"
    fig.savefig(salida, dpi=130)
    print(f"Figura: {salida}")


if __name__ == "__main__":
    main()
