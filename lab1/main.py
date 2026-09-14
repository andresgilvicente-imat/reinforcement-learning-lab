"""Script principal de la Práctica 1 (Aprendizaje por Refuerzo · IMAT).

Ejecuta Policy Iteration y/o Value Iteration sobre el entorno
"Jump To The Goal", imprime V*(s) y pi*(s), y visualiza la política obtenida.

Uso básico (el del enunciado):

    python main.py --env_behavior deterministic --algorithm policy_iteration

Otras opciones útiles para el análisis del informe:

    python main.py --algorithm both --gamma 0.5
    python main.py --algorithm value_iteration --tol 1e-1
    python main.py --env_behavior stochastic --algorithm both
    python main.py --algorithm both --r_s -3
    python main.py --algorithm policy_iteration --start_state 12
    python main.py --algorithm both --no_render      # sin ventana pygame

Los hiperparámetros por defecto (factor de descuento gamma y tolerancia de
convergencia) están definidos justo debajo y pueden sobrescribirse desde la
línea de comandos.
"""

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
from policy_iteration import policy_iteration
from value_iteration import value_iteration

register(id="JumpToTheGoalEnv-v0", entry_point="env:JumpToTheGoalEnv")

# --- Hiperparámetros ------------------------------------------------------
GAMMA = 0.9
CONVERGENCE_TOLERANCE = 1e-3
# Probabilidad de ejecutar la acción elegida en el entorno estocástico
STOCHASTIC_PROB = 0.8
# Pasos máximos de la visualización de un episodio
MAX_ROLLOUT_STEPS = 20
# Segundos entre pasos de la visualización
ROLLOUT_DELAY = 0.4

ACTION_SYMBOL = {0: "/", 1: "\\"}
CELL_WIDTH = 8


def build_parser():
    parser = argparse.ArgumentParser(
        description="Práctica 1: ecuaciones de Bellman, Policy y Value Iteration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env_behavior",
        help="Comportamiento del entorno: determinista o estocástico.",
        choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    parser.add_argument(
        "--algorithm",
        help="Algoritmo a ejecutar.",
        choices=["both", "policy_iteration", "value_iteration"],
        default="both",
    )
    parser.add_argument(
        "--gamma", help="Factor de descuento.", type=float, default=GAMMA
    )
    parser.add_argument(
        "--tol",
        help="Umbral de convergencia (convergence_tolerance).",
        type=float,
        default=CONVERGENCE_TOLERANCE,
    )
    parser.add_argument(
        "--r_s",
        help="Recompensa de una transición entre estados no terminales.",
        type=float,
        default=R_STEP,
    )
    parser.add_argument(
        "--prob",
        help="Probabilidad de éxito de la acción en el entorno estocástico.",
        type=float,
        default=STOCHASTIC_PROB,
    )
    parser.add_argument(
        "--start_state",
        help="Estado inicial del episodio que se visualiza.",
        type=int,
        choices=list(INITIAL_STATES),
        default=INITIAL_STATES[0],
    )
    parser.add_argument(
        "--no_render",
        help="No abrir la ventana de visualización.",
        action="store_true",
    )
    return parser


def state_name(state):
    """Nombre del estado: 'T' para el terminal, 's<n>' para el resto."""
    return "T" if state == TERMINAL_STATE else f"s{state}"


def grid_layout(desc, contenido):
    """Dispone sobre la rejilla 7x5 lo que ``contenido(estado)`` devuelve.

    La rejilla no es el espacio de estados: sirve para leer V* y pi* con la
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
        # Las filas sin estados (todo terminal) no necesitan línea de valores
        if "".join(valores).strip():
            lineas.append("".join(valores).rstrip())
    return "\n".join(lineas)


def print_solution(name, V, policy, desc, elapsed):
    """Imprime V*, pi* y el valor de los estados iniciales."""
    np.set_printoptions(precision=3, suppress=True)
    print(f"\n{'-' * 60}\n{name} — convergencia en {elapsed:.6f} s\n{'-' * 60}")

    print("V*(s) sobre la rejilla:")
    print(grid_layout(desc, lambda s: f"{V[s]:.2f}"))
    print(f"\n  V*(T) = {V[TERMINAL_STATE]:.2f}")

    print("\npi*(s) sobre la rejilla   ( / = arriba-derecha, \\ = abajo-derecha ):")
    print(grid_layout(desc, lambda s: ACTION_SYMBOL[int(policy[s])]))

    print("\nVector completo (s0 ... s19):")
    print(f"  V*  = {V[:N_NON_TERMINAL]}")
    print(f"  pi* = {policy[:N_NON_TERMINAL]}")

    print("\nValor de los estados iniciales:")
    print("  " + "  ".join(f"V*(s{s})={V[s]:7.3f}" for s in INITIAL_STATES))


def run_rollout(env, policy, start_state, max_steps=MAX_ROLLOUT_STEPS):
    """Ejecuta y visualiza un episodio siguiendo ``policy``.

    Devuelve la trayectoria de estados y el retorno sin descontar.
    """
    state, _ = env.reset(options={"initial_state": start_state})
    trajectory = [state]
    total_reward = 0.0
    for _ in range(max_steps):
        state, reward, terminated, truncated, _ = env.step(int(policy[state]))
        trajectory.append(state)
        total_reward += reward
        time.sleep(ROLLOUT_DELAY)
        if terminated or truncated:
            break
    camino = " -> ".join(state_name(s) for s in trajectory)
    print(f"\nEpisodio desde s{start_state}: {camino}")
    print(f"Recompensa acumulada (sin descuento): {total_reward:.2f}")
    return trajectory, total_reward


def main():
    args = build_parser().parse_args()

    deterministic = args.env_behavior == "deterministic"
    render_mode = None if args.no_render else "human"

    env = gym.make(
        "JumpToTheGoalEnv-v0",
        render_mode=render_mode,
        deterministic=deterministic,
        prob=args.prob,
        r_step=args.r_s,
    )

    # El modelo del entorno: P[s][a] -> [(prob, next_state, reward, terminal), ...]
    P = env.unwrapped.P
    nS = env.unwrapped.nS
    nA = env.unwrapped.nA
    desc = env.unwrapped.desc

    print(f"Entorno: {args.env_behavior}", end="")
    if not deterministic:
        print(f" (p_éxito={args.prob}, p_fallo={1 - args.prob:.2f})", end="")
    print(f" | gamma={args.gamma} | tol={args.tol}")
    print(f"Recompensas: r_s={args.r_s} | r_obs={R_OBSTACLE} | r_goal={R_GOAL}")
    print(f"Estados: {N_NON_TERMINAL} visitables (s0-s{N_NON_TERMINAL - 1}) + T = {nS}")

    if args.algorithm in ("both", "policy_iteration"):
        start = time.time()
        V_pi, pi_pi = policy_iteration(P, nS, nA, gamma=args.gamma, tol=args.tol)
        print_solution("POLICY ITERATION", V_pi, pi_pi, desc, time.time() - start)
        if render_mode is not None:
            run_rollout(env, pi_pi, args.start_state)

    if args.algorithm in ("both", "value_iteration"):
        start = time.time()
        V_vi, pi_vi = value_iteration(P, nS, nA, gamma=args.gamma, tol=args.tol)
        print_solution("VALUE ITERATION", V_vi, pi_vi, desc, time.time() - start)
        if render_mode is not None:
            run_rollout(env, pi_vi, args.start_state)

    env.close()


if __name__ == "__main__":
    main()
