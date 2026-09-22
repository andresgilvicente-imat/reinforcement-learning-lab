"""Experimento 1 · Policy Iteration: efecto de gamma y del umbral de
convergencia (theta / tol) sobre el aprendizaje y la convergencia.

======================================================================
PREGUNTA QUE RESPONDE ESTE SCRIPT
----------------------------------------------------------------------
"Analizar el proceso de aprendizaje del agente y realizar un analisis
sobre como se ve afectado el aprendizaje del agente y la convergencia
del mismo para distintos valores de los hiperparametros gamma y umbral
de convergencia theta."
======================================================================

Este script usa UNICAMENTE Policy Iteration (PI). No se llama a Value
Iteration en ningun momento: la comparacion PI vs VI se hace aparte, en
`experiment_vi_vs_pi.py`, que responde a la otra pregunta de la practica.

Que se mide, para el entorno determinista y el estocastico:

  BLOQUE A - Barrido de gamma (tol fija = BASE_TOL)
    - V*(s_init) para cada estado inicial            -> que tan "optimista"
      es el agente segun el horizonte de planificacion.
    - Iteraciones externas de PI y barridos totales
      de evaluacion                                  -> coste de convergencia.
    - Tiempo de convergencia (ms).
    - Composicion de la politica optima (cuantos
      estados eligen cada accion) y cuanto cambia
      respecto a la politica de referencia (gamma=BASE_GAMMA).
    - Resultado practico: tasa de exito de rollouts
      desde cada estado inicial.

  BLOQUE B - Barrido de tol (gamma fija = BASE_GAMMA)
    - Coste de convergencia (iteraciones/barridos) vs tol.
    - Calidad de la solucion: error en V y "regret" de la politica
      devuelta frente a una solucion de referencia (PI con tol muy
      estricta) -> muestra que un theta demasiado laxo NO solo acelera
      la convergencia, tambien puede degradar la politica aprendida.
    - Curvas de residuo (delta) de la evaluacion de la politica optima,
      con las lineas de varios umbrales tol superpuestas -> visualiza
      exactamente en que barrido "corta" cada theta.

  BLOQUE C - Proceso de aprendizaje (paso a paso)
    - Evolucion de V(s_init) iteracion-externa a iteracion-externa de
      PI, para un par de valores de gamma representativos.
    - Numero de estados cuya accion cambia en cada iteracion externa
      (mejora de politica) -> ritmo al que el agente "aprende" la
      politica optima y en que iteracion se estabiliza.

  BLOQUE D - Mapas de calor conjuntos gamma x tol del coste de PI.

r_step, r_obstacle y r_goal se mantienen en los valores por defecto del
enunciado.
"""

import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import (
    GRID_WORLD,
    CELL_TO_STATE,
    INITIAL_STATES,
    N_NON_TERMINAL,
    R_STEP,
    R_OBSTACLE,
    R_GOAL,
    JumpToTheGoalEnv,
)
from policy_iteration import policy_improvement, policy_iteration

# ============================================================================
# Configuracion
# ============================================================================
BASE_GAMMA = 0.9
BASE_TOL = 1e-3

GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
TOLS = [10.0, 3.0, 1.0, 0.3, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

STOCHASTIC_PROB = 0.8
MAX_ROLLOUT_STEPS = 20
N_STOCHASTIC_EPISODES = 300
TIMING_REPEATS = 5
EXACT_TOL = 1e-13  # umbral usado SOLO para construir la solucion de referencia
ERR_FLOOR = 1e-16  # piso para poder dibujar ceros en escala log
UP, DOWN = 0, 1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_pi_hyperparams")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
for d in (DATA_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

KINDS = ("det", "sto")
KIND_LABEL = {"det": "determinista", "sto": f"estocastico (prob={STOCHASTIC_PROB})"}
KIND_COLOR = {"det": "tab:blue", "sto": "tab:orange"}


# ============================================================================
# Policy Iteration instrumentado (mismo algoritmo, pero registrando contadores)
# ============================================================================
def policy_evaluation_instrumented(P, nS, nA, policy, gamma, tol, track_deltas=False):
    V = np.zeros(nS)
    sweeps = 0
    deltas = [] if track_deltas else None
    while True:
        sweeps += 1
        delta = 0.0
        for state in range(nS):
            old_value = V[state]
            action = int(policy[state])
            V[state] = sum(
                p * (r + gamma * V[n]) for p, n, r, _ in P[state][action]
            )
            delta = max(delta, abs(old_value - V[state]))
        if track_deltas:
            deltas.append(delta)
        if delta < tol:
            break
    return (V, sweeps, deltas) if track_deltas else (V, sweeps)


def policy_iteration_instrumented(P, nS, nA, gamma, tol, track_learning=False):
    """Igual que policy_iteration, pero devuelve contadores y, si se pide,
    el rastro completo del proceso de aprendizaje (V y politica en cada
    iteracion externa)."""
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    outer_iterations = 0
    total_eval_sweeps = 0
    history = [] if track_learning else None

    while True:
        outer_iterations += 1
        V, sweeps = policy_evaluation_instrumented(P, nS, nA, policy, gamma, tol)
        total_eval_sweeps += sweeps
        new_policy = policy_improvement(P, nS, nA, V, gamma)

        if track_learning:
            n_changes = int(np.sum(new_policy[:N_NON_TERMINAL] != policy[:N_NON_TERMINAL])) \
                if outer_iterations > 1 else N_NON_TERMINAL
            history.append({
                "outer_iter": outer_iterations,
                "eval_sweeps": sweeps,
                "V_init": {f"s{s}": float(V[s]) for s in INITIAL_STATES},
                "n_policy_changes": n_changes,
                "policy": new_policy.tolist(),
            })

        converged = np.array_equal(policy, new_policy)
        policy = new_policy
        if converged:
            break

    return (V, policy, outer_iterations, total_eval_sweeps, history)


def best_time_ms(fn, *args, repeats=TIMING_REPEATS):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def q_values(P, nS, nA, V, gamma):
    return np.array(
        [[sum(p * (r + gamma * V[n]) for p, n, r, _ in P[s][a]) for a in range(nA)] for s in range(nS)]
    )


def optimal_action_sets(P, nS, nA, V, gamma, eps=1e-9):
    """Conjunto de acciones optimas por estado no terminal (gestiona empates)."""
    Q = q_values(P, nS, nA, V, gamma)
    return [set(np.flatnonzero(Q[s] >= Q[s].max() - eps)) for s in range(N_NON_TERMINAL)]


def policy_regret(P, nS, nA, policy, V_ref, gamma):
    """max_s [V_ref(s) - V^pi(s)] sobre estados no terminales (0 = optima)."""
    V_pi, _ = policy_evaluation_instrumented(P, nS, nA, policy, gamma, EXACT_TOL)
    return float(np.max(V_ref[:N_NON_TERMINAL] - V_pi[:N_NON_TERMINAL]))


# ============================================================================
# Rollouts (solo para el barrido de gamma con tol = BASE_TOL)
# ============================================================================
def rollout_det(env, policy, start, max_steps):
    state, _ = env.reset(options={"initial_state": start})
    total = 0.0
    for _ in range(max_steps):
        state, r, term, _, _ = env.step(int(policy[state]))
        total += r
        if term:
            return {"outcome": "goal" if r == env.r_goal else "obstacle", "reward": total}
    return {"outcome": "timeout", "reward": total}


def rollout_sto(env, policy, start, max_steps, n_ep, seed_base):
    outcomes, rewards = [], []
    for ep in range(n_ep):
        state, _ = env.reset(seed=seed_base + ep, options={"initial_state": start})
        total, out = 0.0, "timeout"
        for _ in range(max_steps):
            state, r, term, _, _ = env.step(int(policy[state]))
            total += r
            if term:
                out = "goal" if r == env.r_goal else "obstacle"
                break
        outcomes.append(out)
        rewards.append(total)
    n = len(outcomes)
    return {
        "success_rate": outcomes.count("goal") / n,
        "obstacle_rate": outcomes.count("obstacle") / n,
        "timeout_rate": outcomes.count("timeout") / n,
        "avg_reward": float(np.mean(rewards)),
    }


# ============================================================================
# Ejecucion de la malla completa (BLOQUES A, B, D)
# ============================================================================
def run_grid():
    envs = {
        "det": JumpToTheGoalEnv(deterministic=True, prob=1.0, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
        "sto": JumpToTheGoalEnv(deterministic=False, prob=STOCHASTIC_PROB, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
    }
    rows = []
    delta_hist = {k: {} for k in KINDS}   # curva de residuo de la politica optima, por gamma
    opt_sets = {k: {} for k in KINDS}     # conjuntos de acciones optimas, por gamma (para empates)

    for kind in KINDS:
        env = envs[kind]
        P, nS, nA = env.P, env.nS, env.nA
        print(f"\n##### entorno {KIND_LABEL[kind]} #####")

        for gamma in GAMMAS:
            # --- solucion de referencia para este gamma: PI con tol muy estricta
            V_ref, pol_ref, _, _, _ = policy_iteration_instrumented(P, nS, nA, gamma, EXACT_TOL)
            _, _, ref_deltas = policy_evaluation_instrumented(
                P, nS, nA, pol_ref, gamma, EXACT_TOL, track_deltas=True
            )
            delta_hist[kind][gamma] = ref_deltas
            opt_sets[kind][gamma] = optimal_action_sets(P, nS, nA, V_ref, gamma)
            print(f"  gamma={gamma}: PI de referencia converge tras {len(ref_deltas)} barridos "
                  f"de evaluacion de la politica optima")

            for tol in TOLS:
                V_pi, pol_pi, pi_outer, pi_eval, _ = policy_iteration_instrumented(P, nS, nA, gamma, tol)

                row = {
                    "env": kind, "gamma": gamma, "tol": tol,
                    "pi_outer": pi_outer,
                    "pi_eval_sweeps": pi_eval,
                    "pi_time_ms": best_time_ms(policy_iteration, P, nS, nA, gamma, tol),
                    "err_V_pi": float(np.max(np.abs(V_pi - V_ref))),
                    "regret_pi": policy_regret(P, nS, nA, pol_pi, V_ref, gamma),
                    "V_init": {f"s{s}": float(V_pi[s]) for s in INITIAL_STATES},
                    "V": V_pi.tolist(),
                    "policy": pol_pi.tolist(),
                }

                if tol == BASE_TOL:  # rollouts solo en la fila base de cada gamma
                    if kind == "det":
                        row["rollouts"] = {
                            f"s{s}": rollout_det(env, pol_pi, s, MAX_ROLLOUT_STEPS)
                            for s in INITIAL_STATES
                        }
                    else:
                        row["rollouts"] = {
                            f"s{s}": rollout_sto(env, pol_pi, s, MAX_ROLLOUT_STEPS,
                                                 N_STOCHASTIC_EPISODES,
                                                 seed_base=10_000 + s * 137 + int(gamma * 1000))
                            for s in INITIAL_STATES
                        }
                rows.append(row)

    return rows, delta_hist, opt_sets


def get(rows, kind, gamma, tol):
    for r in rows:
        if r["env"] == kind and r["gamma"] == gamma and r["tol"] == tol:
            return r
    raise KeyError((kind, gamma, tol))


def gamma_slice(rows, kind):
    return [get(rows, kind, g, BASE_TOL) for g in GAMMAS]


def tol_slice(rows, kind):
    return [get(rows, kind, BASE_GAMMA, t) for t in TOLS]


# ============================================================================
# BLOQUE C: proceso de aprendizaje paso a paso (unos pocos casos representativos)
# ============================================================================
LEARNING_GAMMAS = [0.5, 0.9, 0.99]


def run_learning_traces():
    envs = {
        "det": JumpToTheGoalEnv(deterministic=True, prob=1.0, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
        "sto": JumpToTheGoalEnv(deterministic=False, prob=STOCHASTIC_PROB, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
    }
    traces = {k: {} for k in KINDS}
    for kind in KINDS:
        env = envs[kind]
        for gamma in LEARNING_GAMMAS:
            _, _, _, _, history = policy_iteration_instrumented(
                env.P, env.nS, env.nA, gamma, BASE_TOL, track_learning=True
            )
            traces[kind][gamma] = history
    return traces


# ============================================================================
# Guardado de datos
# ============================================================================
def save_data(rows, delta_hist, traces):
    with open(os.path.join(DATA_DIR, "grid_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(DATA_DIR, "pi_delta_history.json"), "w") as f:
        json.dump({k: {str(g): v for g, v in d.items()} for k, d in delta_hist.items()}, f)
    with open(os.path.join(DATA_DIR, "learning_traces.json"), "w") as f:
        json.dump({k: {str(g): v for g, v in d.items()} for k, d in traces.items()}, f, indent=1)

    cols = ["env", "gamma", "tol", "pi_outer", "pi_eval_sweeps", "pi_time_ms",
            "err_V_pi", "regret_pi"]
    with open(os.path.join(DATA_DIR, "grid_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"\nDatos guardados en: {DATA_DIR}")


# ============================================================================
# Utilidades de dibujo
# ============================================================================
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, name), dpi=150)
    plt.close(fig)


def _mark_base_gamma(ax):
    ax.axvline(BASE_GAMMA, color="gray", linestyle=":", linewidth=1)


def _mark_base_tol(ax):
    ax.axvline(BASE_TOL, color="gray", linestyle=":", linewidth=1)


# ============================================================================
# BLOQUE A: barrido de gamma (tol = BASE_TOL)
# ============================================================================
def plot_gamma_value(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = gamma_slice(rows, kind)
        for s in INITIAL_STATES:
            ax.plot(GAMMAS, [r["V_init"][f"s{s}"] for r in sl], marker="o", label=f"V*(s{s})")
        _mark_base_gamma(ax)
        ax.set_title(KIND_LABEL[kind])
        ax.set_xlabel("gamma")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("V*(estado inicial) segun Policy Iteration")
    axes[0].legend()
    fig.suptitle(f"[PI] V*(s_init) vs gamma (tol={BASE_TOL:g})")
    _save(fig, name)


def plot_gamma_policy_change(rows, opt_sets, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for kind in KINDS:
        ref = opt_sets[kind][BASE_GAMMA]
        diffs = []
        for g in GAMMAS:
            cur = opt_sets[kind][g]
            diffs.append(sum(1 for a, b in zip(ref, cur) if not (a & b)))
        ax.plot(GAMMAS, diffs, marker="o", color=KIND_COLOR[kind], label=KIND_LABEL[kind])
    _mark_base_gamma(ax)
    ax.set_xlabel("gamma")
    ax.set_ylabel(f"estados cuya accion optima difiere de gamma={BASE_GAMMA}")
    ax.set_title("Cambios en la politica optima (PI)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for kind in KINDS:
        sl = gamma_slice(rows, kind)
        n_up = [int(np.sum(np.array(r["policy"])[:N_NON_TERMINAL] == UP)) for r in sl]
        ax.plot(GAMMAS, n_up, marker="o", color=KIND_COLOR[kind], label=KIND_LABEL[kind])
    _mark_base_gamma(ax)
    ax.set_xlabel("gamma")
    ax.set_ylabel(f"estados con accion arriba-derecha (de {N_NON_TERMINAL})")
    ax.set_title("Composicion de la politica (PI)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, name)


def plot_gamma_convergence(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = gamma_slice(rows, kind)
        ax.plot(GAMMAS, [r["pi_eval_sweeps"] for r in sl], marker="s", label="barridos de evaluacion (total)")
        ax.plot(GAMMAS, [r["pi_outer"] for r in sl], marker="o", label="iteraciones externas")
        _mark_base_gamma(ax)
        ax.set_title(KIND_LABEL[kind])
        ax.set_xlabel("gamma")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coste hasta converger (PI)")
    axes[0].legend()
    fig.suptitle(f"[PI] Coste de convergencia vs gamma (tol={BASE_TOL:g})")
    _save(fig, name)


def plot_gamma_time(rows, name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in KINDS:
        sl = gamma_slice(rows, kind)
        ax.plot(GAMMAS, [r["pi_time_ms"] for r in sl], marker="o", color=KIND_COLOR[kind],
                label=KIND_LABEL[kind])
    _mark_base_gamma(ax)
    ax.set_xlabel("gamma")
    ax.set_ylabel("tiempo de convergencia de PI (ms)")
    ax.set_title(f"[PI] Tiempo de convergencia vs gamma (tol={BASE_TOL:g})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, name)


def plot_gamma_outcomes(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    sl = gamma_slice(rows, "det")
    n_goal = [sum(1 for s in INITIAL_STATES if r["rollouts"][f"s{s}"]["outcome"] == "goal") for r in sl]
    ax.plot(GAMMAS, n_goal, marker="o", color="darkgreen")
    ax.set_ylim(-0.3, len(INITIAL_STATES) + 0.3)
    ax.set_yticks(range(len(INITIAL_STATES) + 1))
    _mark_base_gamma(ax)
    ax.set_xlabel("gamma")
    ax.set_ylabel(f"estados iniciales (de {len(INITIAL_STATES)}) que llegan al objetivo")
    ax.set_title("Determinista")
    ax.grid(alpha=0.3)

    ax = axes[1]
    sl = gamma_slice(rows, "sto")
    for s in INITIAL_STATES:
        ax.plot(GAMMAS, [r["rollouts"][f"s{s}"]["success_rate"] for r in sl], marker="o", label=f"s{s}")
    _mark_base_gamma(ax)
    ax.set_xlabel("gamma")
    ax.set_ylabel("tasa de exito")
    ax.set_title(f"Estocastico ({N_STOCHASTIC_EPISODES} episodios/estado)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("[PI] Resultado practico de la politica aprendida vs gamma")
    _save(fig, name)


def plot_gamma_policy_grids(rows, name):
    nrow, ncol = len(GRID_WORLD), len(GRID_WORLD[0])
    ncols = 3
    nrows = int(np.ceil(len(GAMMAS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)
    arrow = {UP: (0.32, 0.32), DOWN: (0.32, -0.32)}
    for ax, g in zip(axes, GAMMAS):
        r = get(rows, "det", g, BASE_TOL)
        V, pol = np.array(r["V"]), np.array(r["policy"])
        heat = np.full((nrow, ncol), np.nan)
        for cell, st in CELL_TO_STATE.items():
            heat[divmod(cell, ncol)] = V[st]
        ax.imshow(heat, cmap="RdYlGn", vmin=np.nanmin(heat), vmax=np.nanmax(heat))
        for row_i, line in enumerate(GRID_WORLD):
            for col_i, letter in enumerate(line):
                if letter == "X":
                    ax.add_patch(plt.Rectangle((col_i - 0.5, row_i - 0.5), 1, 1, color="black"))
                elif letter == "G":
                    ax.text(col_i, row_i, "GOAL", ha="center", va="center", fontsize=7,
                            fontweight="bold", color="white")
                else:
                    st = CELL_TO_STATE[row_i * ncol + col_i]
                    dx, dy = arrow[int(pol[st])]
                    ax.annotate("", xy=(col_i + dx, row_i - dy),
                                xytext=(col_i - dx * 0.2, row_i + dy * 0.2),
                                arrowprops=dict(arrowstyle="->", color="black", lw=1.6))
                    ax.text(col_i, row_i + 0.32, f"s{st}\n{V[st]:.1f}", ha="center",
                            va="center", fontsize=5.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"gamma = {g:g}", fontsize=10)
    for ax in axes[len(GAMMAS):]:
        ax.axis("off")
    fig.suptitle("[PI] Politica optima determinista - comparacion por gamma", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOTS_DIR, name), dpi=140)
    plt.close(fig)


# ============================================================================
# BLOQUE B: barrido de tolerancia (gamma = BASE_GAMMA)
# ============================================================================
def plot_tol_sweeps(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = tol_slice(rows, kind)
        ax.plot(TOLS, [r["pi_eval_sweeps"] for r in sl], marker="s", label="barridos de evaluacion (total)")
        ax.plot(TOLS, [r["pi_outer"] for r in sl], marker="o", label="iteraciones externas")
        ax.set_xscale("log")
        _mark_base_tol(ax)
        ax.set_title(f"{KIND_LABEL[kind]} (gamma={BASE_GAMMA})")
        ax.set_xlabel("tolerancia (theta)")
        ax.grid(alpha=0.3)
        if kind == "det":
            ax.legend()
    axes[0].set_ylabel("coste hasta converger (PI)")
    fig.suptitle("[PI] Coste de convergencia vs tolerancia")
    _save(fig, name)


def plot_tol_accuracy(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    bound = [BASE_GAMMA * t / (1 - BASE_GAMMA) for t in TOLS]

    ax = axes[0]
    for kind in KINDS:
        sl = tol_slice(rows, kind)
        ax.plot(TOLS, [max(r["err_V_pi"], ERR_FLOOR) for r in sl], marker="s",
                color=KIND_COLOR[kind], label=KIND_LABEL[kind])
    ax.plot(TOLS, bound, color="gray", linestyle=":", label="cota gamma*tol/(1-gamma)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _mark_base_tol(ax)
    ax.set_xlabel("tolerancia (theta)")
    ax.set_ylabel(f"max |V_PI - V_referencia|   (ceros dibujados en {ERR_FLOOR:g})")
    ax.set_title("Error en la funcion de valor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for kind in KINDS:
        sl = tol_slice(rows, kind)
        ax.plot(TOLS, [max(r["regret_pi"], ERR_FLOOR) for r in sl], marker="s",
                color=KIND_COLOR[kind], label=KIND_LABEL[kind])
    ax.set_xscale("log")
    ax.set_yscale("log")
    _mark_base_tol(ax)
    ax.set_xlabel("tolerancia (theta)")
    ax.set_ylabel(f"regret maximo de la politica  (0 -> {ERR_FLOOR:g})")
    ax.set_title("Calidad de la politica devuelta")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("[PI] Un theta laxo acelera la convergencia... a costa de precision")
    _save(fig, name)


def plot_delta_curves(delta_hist, name):
    gammas_shown = [0.5, 0.9, 0.99, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for g in gammas_shown:
            d = np.array(delta_hist[kind][g])
            k = np.arange(1, len(d) + 1)
            pos = d > 0
            line, = ax.plot(k[pos], d[pos], marker="o", label=f"gamma={g:g}")
            if (~pos).any():
                ax.plot(k[~pos][:1], [1e-14], marker="X", color=line.get_color(), markersize=9)
        for t in (1.0, 1e-3, 1e-6):
            ax.axhline(t, color="gray", linestyle=":", linewidth=1)
            ax.text(ax.get_xlim()[1], t, f" tol={t:g}", va="bottom", ha="right", fontsize=7, color="gray")
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 50)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("barrido de evaluacion de la politica optima")
        ax.set_title(KIND_LABEL[kind])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("delta (maximo cambio en el barrido)")
    axes[0].legend()
    fig.suptitle("[PI] Curva de residuo: theta decide en que barrido se corta (X = delta exactamente 0)")
    _save(fig, name)


# ============================================================================
# BLOQUE C: proceso de aprendizaje paso a paso
# ============================================================================
def plot_learning_curves(traces, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for g in LEARNING_GAMMAS:
            hist = traces[kind][g]
            iters = [h["outer_iter"] for h in hist]
            v0 = [h["V_init"][f"s{INITIAL_STATES[0]}"] for h in hist]
            ax.plot(iters, v0, marker="o", label=f"gamma={g:g}")
        ax.set_xlabel("iteracion externa de PI (evaluacion + mejora)")
        ax.set_title(KIND_LABEL[kind])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"V(s{INITIAL_STATES[0]}) tras cada iteracion externa")
    axes[0].legend()
    fig.suptitle("[PI] Evolucion del valor aprendido a lo largo del proceso de aprendizaje")
    _save(fig, name)


def plot_policy_changes(traces, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for g in LEARNING_GAMMAS:
            hist = traces[kind][g]
            iters = [h["outer_iter"] for h in hist]
            changes = [h["n_policy_changes"] for h in hist]
            ax.plot(iters, changes, marker="o", label=f"gamma={g:g}")
        ax.set_xlabel("iteracion externa de PI")
        ax.set_title(KIND_LABEL[kind])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"estados que cambian de accion (de {N_NON_TERMINAL})")
    axes[0].legend()
    fig.suptitle("[PI] Ritmo de aprendizaje: cuantos estados cambian de accion en cada mejora de politica")
    _save(fig, name)


# ============================================================================
# BLOQUE D: mapas de calor gamma x tol
# ============================================================================
def _heatmap(ax, rows, kind, key, title, log_floor=False, cmap="viridis"):
    Z = np.array([[get(rows, kind, g, t)[key] for t in TOLS] for g in GAMMAS], dtype=float)
    shown = np.log10(np.maximum(Z, ERR_FLOOR)) if log_floor else Z
    im = ax.imshow(shown, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(TOLS)))
    ax.set_xticklabels([f"{t:g}" for t in TOLS], rotation=45, fontsize=7)
    ax.set_yticks(range(len(GAMMAS)))
    ax.set_yticklabels([f"{g:g}" for g in GAMMAS], fontsize=8)
    for i in range(len(GAMMAS)):
        for j in range(len(TOLS)):
            v = Z[i, j]
            txt = ("0" if v == 0 else f"{np.log10(v):.0f}") if log_floor else f"{int(v)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("tolerancia (theta)")
    ax.set_ylabel("gamma")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def plot_heatmaps(rows, name):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _heatmap(axes[0, 0], rows, "det", "pi_eval_sweeps", "PI barridos de evaluacion - determinista")
    _heatmap(axes[0, 1], rows, "sto", "pi_eval_sweeps", "PI barridos de evaluacion - estocastico")
    _heatmap(axes[1, 0], rows, "det", "regret_pi", "log10 regret de la politica - determinista",
             log_floor=True, cmap="magma")
    _heatmap(axes[1, 1], rows, "sto", "regret_pi", "log10 regret de la politica - estocastico",
             log_floor=True, cmap="magma")
    fig.suptitle("[PI] Coste y precision en la malla gamma x theta")
    _save(fig, name)


def make_all_plots(rows, delta_hist, opt_sets, traces):
    plot_gamma_value(rows, "A1_gamma_value.png")
    plot_gamma_policy_change(rows, opt_sets, "A2_gamma_policy_change.png")
    plot_gamma_convergence(rows, "A3_gamma_convergence.png")
    plot_gamma_time(rows, "A4_gamma_time.png")
    plot_gamma_outcomes(rows, "A5_gamma_outcomes.png")
    plot_gamma_policy_grids(rows, "A6_gamma_policy_grids.png")

    plot_tol_sweeps(rows, "B1_tol_sweeps.png")
    plot_tol_accuracy(rows, "B2_tol_accuracy.png")
    plot_delta_curves(delta_hist, "B3_tol_delta_curves.png")

    plot_learning_curves(traces, "C1_learning_curves.png")
    plot_policy_changes(traces, "C2_policy_changes_per_iteration.png")

    plot_heatmaps(rows, "D1_joint_heatmaps.png")
    print(f"Graficas guardadas en: {PLOTS_DIR}")


if __name__ == "__main__":
    print("Ejecutando barrido de gamma y tolerancia (SOLO Policy Iteration)...")
    print(f"GAMMAS={GAMMAS}\nTOLS={TOLS}\nbase: gamma={BASE_GAMMA}, tol={BASE_TOL:g} | "
          f"r_step={R_STEP}, r_obs={R_OBSTACLE}, r_goal={R_GOAL}")
    rows, delta_hist, opt_sets = run_grid()
    traces = run_learning_traces()
    save_data(rows, delta_hist, traces)
    make_all_plots(rows, delta_hist, opt_sets, traces)
    print("\nListo. Revisa 'results_pi_hyperparams/' (datos en 'data/', graficas en 'plots/').")
