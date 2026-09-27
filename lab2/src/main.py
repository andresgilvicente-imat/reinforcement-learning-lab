"""Script principal de la Práctica 2 (Aprendizaje por Refuerzo · IMAT).

Entrena un agente con SARSA y/o Q-learning sobre el entorno "Jump To The Goal"
—el mismo de la Práctica 1— e imprime la tabla Q aprendida, la política greedy
y un resumen del entrenamiento.

Uso básico:

    python main.py --algorithm sarsa
    python main.py --algorithm q_learning --episodes 20000

Opciones útiles para el informe:

    python main.py --algorithm both --episodes 20000 --alpha 0.5
    python main.py --algorithm q_learning --q_init 5          # Q_0 optimista
    python main.py --algorithm sarsa --epsilon 0.1
    python main.py --algorithm both --r_s -1 --r_obs -5 --r_goal 5
    python main.py --algorithm q_learning --graficas          # las dos curvas
    python main.py --algorithm sarsa --render                 # ver un episodio

Los hiperparámetros por defecto están definidos justo debajo y pueden
sobrescribirse desde la línea de comandos: no hace falta editar este fichero.
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from env import (
    CELL_TO_STATE,
    INITIAL_STATES,
    N_NON_TERMINAL,
    R_GOAL,
    R_OBSTACLE,
    R_STEP,
    TERMINAL_STATE,
)
from q_learning import q_learning
from sarsa import greedy_policy, sarsa

register(id="JumpToTheGoalEnv-v0", entry_point="env:JumpToTheGoalEnv")

# --- Hiperparámetros ------------------------------------------------------
NUM_EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.9
# epsilon del entrenamiento; en la evaluación se usa la política greedy
EPSILON = 0.2
Q_INIT = 0.0
# Estados cuya evolución de Q(s, a) se registra y se dibuja
TRACK_STATES = (6, 13)
# Episodios sobre los que se promedia el resumen final
VENTANA_RESUMEN = 200
# Pasos máximos y ritmo de la visualización de un episodio
MAX_ROLLOUT_STEPS = 20
ROLLOUT_DELAY = 0.4

ACTION_NAME = {0: "arriba-derecha", 1: "abajo-derecha"}
ACTION_SYMBOL = {0: "/", 1: "\\"}
CELL_WIDTH = 8


def build_parser():
    parser = argparse.ArgumentParser(
        description="Práctica 2: SARSA y Q-learning tabulares.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        help="Algoritmo a entrenar.",
        choices=["both", "sarsa", "q_learning"],
        default="both",
    )
    parser.add_argument(
        "--episodes",
        help="Número de episodios de entrenamiento.",
        type=int,
        default=NUM_EPISODES,
    )
    parser.add_argument(
        "--alpha", help="Tasa de aprendizaje.", type=float, default=ALPHA
    )
    parser.add_argument(
        "--gamma", help="Factor de descuento.", type=float, default=GAMMA
    )
    parser.add_argument(
        "--epsilon",
        help="Probabilidad de explorar durante el entrenamiento.",
        type=float,
        default=EPSILON,
    )
    parser.add_argument(
        "--q_init",
        help="Valor inicial de la tabla Q (Q_0).",
        type=float,
        default=Q_INIT,
    )
    parser.add_argument(
        "--r_s",
        help="Recompensa de una transición entre estados no terminales.",
        type=float,
        default=R_STEP,
    )
    parser.add_argument(
        "--r_obs",
        help="Recompensa al chocar con un obstáculo.",
        type=float,
        default=R_OBSTACLE,
    )
    parser.add_argument(
        "--r_goal",
        help="Recompensa al alcanzar el objetivo.",
        type=float,
        default=R_GOAL,
    )
    parser.add_argument(
        "--seed", help="Semilla del entrenamiento.", type=int, default=0
    )
    parser.add_argument(
        "--track",
        help="Estados cuya evolución de Q se registra, separados por comas.",
        type=str,
        default=",".join(str(s) for s in TRACK_STATES),
    )
    parser.add_argument(
        "--graficas",
        help="Dibujar las curvas de aprendizaje (necesita matplotlib).",
        action="store_true",
    )
    parser.add_argument(
        "--render",
        help="Visualizar un episodio con la política aprendida.",
        action="store_true",
    )
    return parser


def state_name(state):
    """Nombre del estado: 'T' para el terminal, 's<n>' para el resto."""
    return "T" if state == TERMINAL_STATE else f"s{state}"


def grid_layout(desc, contenido):
    """Dispone sobre la rejilla 7x5 lo que ``contenido(estado)`` devuelve.

    La rejilla no es el espacio de estados: sirve para leer la tabla con la
    misma disposición espacial que la figura del enunciado. Las celdas
    terminales se marcan con T.
    """
    nrow, ncol = desc.shape
    lineas = []
    for row in range(nrow):
        etiquetas, valores = [], []
        for col in range(ncol):
            cell = row * ncol + col
            if cell in CELL_TO_STATE:
                estado = CELL_TO_STATE[cell]
                etiquetas.append(f"s{estado}".center(CELL_WIDTH))
                valores.append(contenido(estado).center(CELL_WIDTH))
            else:
                etiquetas.append("T".center(CELL_WIDTH))
                valores.append("".center(CELL_WIDTH))
        lineas.append("".join(etiquetas).rstrip())
        if "".join(valores).strip():
            lineas.append("".join(valores).rstrip())
    return "\n".join(lineas)


def print_resultado(nombre, resultado, desc, elapsed, ventana=VENTANA_RESUMEN):
    """Imprime la tabla Q aprendida, la política greedy y el resumen."""
    Q = resultado["Q"]
    politica = greedy_policy(Q)
    ventana = min(ventana, len(resultado["retornos"]))

    print(f"\n{'-' * 60}\n{nombre} — {elapsed:.2f} s\n{'-' * 60}")

    print("max_a Q(s, a) sobre la rejilla:")
    print(grid_layout(desc, lambda s: f"{np.max(Q[s]):.2f}"))

    print("\nPolítica greedy   ( / = arriba-derecha, \\ = abajo-derecha ):")
    print(grid_layout(desc, lambda s: ACTION_SYMBOL[int(politica[s])]))

    print("\nQ(s, a) de los estados seguidos:")
    for s in resultado["track_states"]:
        valores = "   ".join(
            f"Q({state_name(s)}, {ACTION_NAME[a]}) = {Q[s, a]:+8.4f}"
            for a in range(Q.shape[1])
        )
        print(f"  {valores}")

    exitos = resultado["exitos"]
    print(f"\nÚltimos {ventana} episodios:")
    print(
        f"  retorno medio            = {np.mean(resultado['retornos'][-ventana:]):+.3f}"
    )
    print(f"  pasos medios             = {np.mean(resultado['pasos'][-ventana:]):.2f}")
    print(f"  episodios en el objetivo = {100 * np.mean(exitos[-ventana:]):.1f} %")
    print(f"  pasos totales            = {resultado['pasos_acumulados'][-1]}")


def dibuja_curvas(resultados, ruta="figuras"):
    """Dibuja las dos curvas que sugiere el enunciado, por algoritmo.

    1. Episodios exitosos acumulados frente al número de pasos.
    2. Q(s, a) de los estados seguidos frente al número de pasos.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    Path(ruta).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nombre, resultado in resultados.items():
        ax.plot(
            resultado["pasos_acumulados"],
            np.cumsum(resultado["exitos"]),
            label=nombre,
        )
    ax.set_xlabel("pasos de entrenamiento")
    ax.set_ylabel("episodios acabados en el objetivo (acumulados)")
    ax.set_title("Episodios exitosos frente a pasos")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    destino = Path(ruta) / "episodios_exitosos.png"
    fig.savefig(destino, dpi=150)
    print(f"\nGuardado {destino}")

    fig, axes = plt.subplots(
        1, len(resultados), figsize=(6 * len(resultados), 4.5), squeeze=False
    )
    for ax, (nombre, resultado) in zip(axes[0], resultados.items(), strict=True):
        for i, s in enumerate(resultado["track_states"]):
            for a in range(resultado["q_history"].shape[2]):
                ax.plot(
                    resultado["pasos_acumulados"],
                    resultado["q_history"][:, i, a],
                    label=f"Q({state_name(s)}, {ACTION_NAME[a]})",
                )
        ax.set_xlabel("pasos de entrenamiento")
        ax.set_ylabel("Q(s, a)")
        ax.set_title(nombre)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    destino = Path(ruta) / "evolucion_q.png"
    fig.savefig(destino, dpi=150)
    print(f"Guardado {destino}")

    plt.show()


def run_rollout(env, politica, start_state, max_steps=MAX_ROLLOUT_STEPS):
    """Ejecuta y visualiza un episodio siguiendo la política greedy."""
    state, _ = env.reset(options={"initial_state": start_state})
    trayectoria = [state]
    retorno = 0.0
    for _ in range(max_steps):
        state, reward, terminated, truncated, _ = env.step(int(politica[state]))
        trayectoria.append(state)
        retorno += reward
        time.sleep(ROLLOUT_DELAY)
        if terminated or truncated:
            break
    camino = " -> ".join(state_name(s) for s in trayectoria)
    print(f"\nEpisodio desde s{start_state}: {camino}")
    print(f"Recompensa acumulada (sin descuento): {retorno:.2f}")
    return trayectoria, retorno


def main():
    args = build_parser().parse_args()
    track_states = tuple(int(s) for s in args.track.split(",") if s.strip())

    env = gym.make(
        "JumpToTheGoalEnv-v0",
        deterministic=True,
        r_step=args.r_s,
        r_obstacle=args.r_obs,
        r_goal=args.r_goal,
        initial_state=None,  # el enunciado pide estado inicial aleatorio
    )
    desc = env.unwrapped.desc

    print(f"Episodios={args.episodes} | alpha={args.alpha} | gamma={args.gamma}")
    print(f"epsilon={args.epsilon} | Q_0={args.q_init} | semilla={args.seed}")
    print(f"Recompensas: r_s={args.r_s} | r_obs={args.r_obs} | r_goal={args.r_goal}")
    print(f"s_init = {{{', '.join(f's{s}' for s in INITIAL_STATES)}}} (aleatorio)")
    print(f"Estados: {N_NON_TERMINAL} visitables (s0-s{N_NON_TERMINAL - 1}) + T")

    algoritmos = {"SARSA": sarsa, "Q-LEARNING": q_learning}
    if args.algorithm == "sarsa":
        algoritmos.pop("Q-LEARNING")
    elif args.algorithm == "q_learning":
        algoritmos.pop("SARSA")

    resultados = {}
    for nombre, algoritmo in algoritmos.items():
        start = time.time()
        resultado = algoritmo(
            env,
            args.episodes,
            args.alpha,
            args.gamma,
            args.epsilon,
            q_init=args.q_init,
            seed=args.seed,
            track_states=track_states,
        )
        print_resultado(nombre, resultado, desc, time.time() - start)
        resultados[nombre] = resultado

    if args.graficas:
        dibuja_curvas(resultados)

    if args.render:
        entorno_visual = gym.make(
            "JumpToTheGoalEnv-v0",
            render_mode="human",
            deterministic=True,
            r_step=args.r_s,
            r_obstacle=args.r_obs,
            r_goal=args.r_goal,
        )
        for nombre, resultado in resultados.items():
            print(f"\n{nombre}: política greedy aprendida")
            run_rollout(
                entorno_visual,
                greedy_policy(resultado["Q"]),
                INITIAL_STATES[0],
            )
        entorno_visual.close()

    env.close()


if __name__ == "__main__":
    main()
