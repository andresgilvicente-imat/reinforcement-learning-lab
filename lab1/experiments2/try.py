"""Experimentos: influencia de r_step (r_s) sobre Policy Iteration y Value
Iteration en el entorno "Jump To The Goal".
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
    STATE_TO_CELL,
    INITIAL_STATES,
    N_NON_TERMINAL,
    TERMINAL_STATE,
    R_OBSTACLE,
    R_GOAL,
    JumpToTheGoalEnv,
)
from policy_iteration import policy_improvement, policy_iteration
from value_iteration import value_iteration

# ============================================================================
# Hiperparametros del analisis (solo r_step varia entre configuraciones)
# ============================================================================
GAMMA = 0.9
CONVERGENCE_TOLERANCE = 1e-3
STOCHASTIC_PROB = 0.8
MAX_ROLLOUT_STEPS = 20
ROLLOUT_DELAY = 0.4
N_STOCHASTIC_EPISODES = 300

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_2")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
GRIDS_DIR = os.path.join(PLOTS_DIR, "policy_grids")
for d in (DATA_DIR, PLOTS_DIR, GRIDS_DIR):
    os.makedirs(d, exist_ok=True)

R_STEP_EXPERIMENTS = [
    {"value": 1.0, "category": "outside", "label": "r_s = 1.0  (fuera, > 0)"},
    {"value": 0.0, "category": "boundary", "label": "r_s = 0.0  (frontera superior)"},
    {"value": -0.5, "category": "inside", "label": "r_s = -0.5 (dentro)"},
    {"value": -1.0, "category": "inside", "label": "r_s = -1.0 (dentro)"},
    {"value": -1.5, "category": "inside", "label": "r_s = -1.5 (dentro)"},
    {"value": -2.0, "category": "inside", "label": "r_s = -2.0 (dentro)"},
    {"value": -2.5, "category": "boundary", "label": "r_s = -2.5 (frontera inferior)"},
    {"value": -5.0, "category": "outside", "label": "r_s = -5.0 (fuera, < -2.5)"},
]

ACTION_SYMBOL = {0: "up-right", 1: "down-right"}
ACTION_ARROW = {0: (0.32, 0.32), 1: (0.32, -0.32)}


def policy_evaluation_instrumented(P, nS, nA, policy, gamma, tol):
    V = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for state in range(nS):
            old_value = V[state]
            action = int(policy[state])
            V[state] = sum(
                probability * (reward + gamma * V[next_state])
                for probability, next_state, reward, terminal in P[state][action]
            )
            delta = max(delta, abs(old_value - V[state]))
        if delta < tol:
            break
    return V, sweeps


def policy_iteration_instrumented(P, nS, nA, gamma, tol):
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    outer_iterations = 0
    total_eval_sweeps = 0
    while True:
        outer_iterations += 1
        V, sweeps = policy_evaluation_instrumented(P, nS, nA, policy, gamma, tol)
        total_eval_sweeps += sweeps
        new_policy = policy_improvement(P, nS, nA, V, gamma)
        converged = np.array_equal(policy, new_policy)
        policy = new_policy
        if converged:
            break
    return V, policy, outer_iterations, total_eval_sweeps


def value_iteration_instrumented(P, nS, nA, gamma, tol):
    V = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for state in range(nS):
            old_value = V[state]
            V[state] = max(
                sum(
                    probability * (reward + gamma * V[next_state])
                    for probability, next_state, reward, terminal in P[state][action]
                )
                for action in range(nA)
            )
            delta = max(delta, abs(old_value - V[state]))
        if delta < tol:
            break
    policy = np.array(
        [
            max(
                range(nA),
                key=lambda action: sum(
                    probability * (reward + gamma * V[next_state])
                    for probability, next_state, reward, terminal in P[state][action]
                ),
            )
            for state in range(nS)
        ]
    )
    return V, policy, sweeps


def rollout_deterministic(env, policy, start_state, max_steps):
    state, _ = env.reset(options={"initial_state": start_state})
    trajectory = [state]
    total_reward = 0.0
    outcome = "timeout"
    for _ in range(max_steps):
        state, reward, terminated, truncated, _ = env.step(int(policy[state]))
        trajectory.append(state)
        total_reward += reward
        if terminated:
            outcome = "goal" if reward == env.r_goal else "obstacle"
            break
    return {
        "trajectory": trajectory,
        "steps": len(trajectory) - 1,
        "total_reward": total_reward,
        "outcome": outcome,
    }


def rollout_stochastic_stats(env, policy, start_state, max_steps, n_episodes, seed_base):
    steps_list, reward_list, outcomes = [], [], []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_base + ep, options={"initial_state": start_state})
        steps = 0
        total_reward = 0.0
        outcome = "timeout"
        for _ in range(max_steps):
            state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            steps += 1
            total_reward += reward
            if terminated:
                outcome = "goal" if reward == env.r_goal else "obstacle"
                break
        steps_list.append(steps)
        reward_list.append(total_reward)
        outcomes.append(outcome)

    n = len(outcomes)
    return {
        "success_rate": outcomes.count("goal") / n,
        "obstacle_rate": outcomes.count("obstacle") / n,
        "timeout_rate": outcomes.count("timeout") / n,
        "avg_steps": float(np.mean(steps_list)),
        "avg_reward": float(np.mean(reward_list)),
        "std_reward": float(np.std(reward_list)),
    }


def plot_policy_grid(ax, V, policy, r_step, title_extra=""):
    nrow, ncol = len(GRID_WORLD), len(GRID_WORLD[0])
    heat = np.full((nrow, ncol), np.nan)

    for cell, state in CELL_TO_STATE.items():
        row, col = divmod(cell, ncol)
        heat[row, col] = V[state]

    vmax = np.nanmax(heat) if np.any(~np.isnan(heat)) else 1.0
    vmin = np.nanmin(heat) if np.any(~np.isnan(heat)) else -1.0
    im = ax.imshow(heat, cmap="RdYlGn", vmin=vmin, vmax=vmax)

    for row, line in enumerate(GRID_WORLD):
        for col, letter in enumerate(line):
            cell = row * ncol + col
            if letter == "X":
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="black"))
            elif letter == "G":
                ax.text(col, row, "GOAL", ha="center", va="center", fontsize=7,
                         fontweight="bold", color="white")
            elif letter == "F":
                state = CELL_TO_STATE[cell]
                action = int(policy[state])
                dx, dy = ACTION_ARROW[action]
                ax.annotate(
                    "", xy=(col + dx, row + (-dy)), xytext=(col - dx * 0.2, row + dy * 0.2),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
                )
                ax.text(col, row + 0.32, f"s{state}\n{V[state]:.1f}", ha="center",
                         va="center", fontsize=5.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"r_s = {r_step:g}{title_extra}", fontsize=10)
    return im


def save_policy_grid_figure(V, policy, r_step, label, path):
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    im = plot_policy_grid(ax, V, policy, r_step)
    fig.suptitle(f"Politica optima (determinista) - {label}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V*(s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_policy_grid_overview(all_rows, path):
    n = len(all_rows)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, row in enumerate(all_rows):
        plot_policy_grid(axes[i], np.array(row["V"]), np.array(row["policy"]), row["r_step"])
    for j in range(len(all_rows), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Politica optima determinista - comparacion por r_s", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_all_experiments():
    results_det = []
    results_sto = []

    for cfg in R_STEP_EXPERIMENTS:
        r_step = cfg["value"]
        print(f"\n=== r_step = {r_step} ({cfg['category']}) ===")

        env_det = JumpToTheGoalEnv(
            render_mode=None, deterministic=True, prob=1.0,
            r_step=r_step, r_obstacle=R_OBSTACLE, r_goal=R_GOAL,
            initial_state=INITIAL_STATES[0],
        )
        P, nS, nA = env_det.P, env_det.nS, env_det.nA

        t0 = time.perf_counter()
        V_pi, pi_pi = policy_iteration(P, nS, nA, gamma=GAMMA, tol=CONVERGENCE_TOLERANCE)
        t_pi = time.perf_counter() - t0
        _, _, pi_outer_iters, pi_eval_sweeps = policy_iteration_instrumented(
            P, nS, nA, GAMMA, CONVERGENCE_TOLERANCE
        )

        t0 = time.perf_counter()
        V_vi, pi_vi = value_iteration(P, nS, nA, gamma=GAMMA, tol=CONVERGENCE_TOLERANCE)
        t_vi = time.perf_counter() - t0
        _, _, vi_sweeps = value_iteration_instrumented(P, nS, nA, GAMMA, CONVERGENCE_TOLERANCE)

        policies_match = bool(np.array_equal(pi_pi, pi_vi))
        max_v_diff = float(np.max(np.abs(V_pi - V_vi)))

        rollouts = {
            f"s{s}": rollout_deterministic(env_det, pi_pi, s, MAX_ROLLOUT_STEPS)
            for s in INITIAL_STATES
        }

        row_det = {
            "r_step": r_step,
            "category": cfg["category"],
            "label": cfg["label"],
            "V": V_pi.tolist(),
            "policy": pi_pi.tolist(),
            "policies_match_pi_vi": policies_match,
            "max_V_diff_pi_vs_vi": max_v_diff,
            "pi_time_s": t_pi,
            "vi_time_s": t_vi,
            "pi_outer_iterations": pi_outer_iters,
            "pi_total_eval_sweeps": pi_eval_sweeps,
            "vi_sweeps": vi_sweeps,
            "V_initial_states": {f"s{s}": float(V_pi[s]) for s in INITIAL_STATES},
            "rollouts": rollouts,
        }
        results_det.append(row_det)

        save_policy_grid_figure(
            V_pi, pi_pi, r_step, cfg["label"],
            os.path.join(GRIDS_DIR, f"policy_grid_r_step_{r_step:g}.png"),
        )

        print(f"  [determinista] PI: {pi_outer_iters} iter. externas / "
              f"{pi_eval_sweeps} barridos de evaluacion, {t_pi * 1000:.2f} ms")
        print(f"  [determinista] VI: {vi_sweeps} barridos, {t_vi * 1000:.2f} ms")
        print(f"  [determinista] pi_PI == pi_VI: {policies_match}")

        env_sto = JumpToTheGoalEnv(
            render_mode=None, deterministic=False, prob=STOCHASTIC_PROB,
            r_step=r_step, r_obstacle=R_OBSTACLE, r_goal=R_GOAL,
            initial_state=INITIAL_STATES[0],
        )
        Ps, nSs, nAs = env_sto.P, env_sto.nS, env_sto.nA

        t0 = time.perf_counter()
        V_pi_s, pi_pi_s = policy_iteration(Ps, nSs, nAs, gamma=GAMMA, tol=CONVERGENCE_TOLERANCE)
        t_pi_s = time.perf_counter() - t0
        _, _, pi_outer_iters_s, pi_eval_sweeps_s = policy_iteration_instrumented(
            Ps, nSs, nAs, GAMMA, CONVERGENCE_TOLERANCE
        )

        t0 = time.perf_counter()
        V_vi_s, pi_vi_s = value_iteration(Ps, nSs, nAs, gamma=GAMMA, tol=CONVERGENCE_TOLERANCE)
        t_vi_s = time.perf_counter() - t0
        _, _, vi_sweeps_s = value_iteration_instrumented(Ps, nSs, nAs, GAMMA, CONVERGENCE_TOLERANCE)

        policies_match_s = bool(np.array_equal(pi_pi_s, pi_vi_s))
        max_v_diff_s = float(np.max(np.abs(V_pi_s - V_vi_s)))

        rollout_stats = {
            f"s{s}": rollout_stochastic_stats(
                env_sto, pi_pi_s, s, MAX_ROLLOUT_STEPS,
                n_episodes=N_STOCHASTIC_EPISODES,
                seed_base=10_000 + s * 137 + int(round(r_step * 100)),
            )
            for s in INITIAL_STATES
        }

        row_sto = {
            "r_step": r_step,
            "category": cfg["category"],
            "label": cfg["label"],
            "V": V_pi_s.tolist(),
            "policy": pi_pi_s.tolist(),
            "policies_match_pi_vi": policies_match_s,
            "max_V_diff_pi_vs_vi": max_v_diff_s,
            "pi_time_s": t_pi_s,
            "vi_time_s": t_vi_s,
            "pi_outer_iterations": pi_outer_iters_s,
            "pi_total_eval_sweeps": pi_eval_sweeps_s,
            "vi_sweeps": vi_sweeps_s,
            "V_initial_states": {f"s{s}": float(V_pi_s[s]) for s in INITIAL_STATES},
            "rollout_stats": rollout_stats,
        }
        results_sto.append(row_sto)

        print(f"  [estocastico]  PI: {pi_outer_iters_s} iter. externas / "
              f"{pi_eval_sweeps_s} barridos de evaluacion, {t_pi_s * 1000:.2f} ms")
        print(f"  [estocastico]  VI: {vi_sweeps_s} barridos, {t_vi_s * 1000:.2f} ms")

        env_det.close()
        env_sto.close()

    return results_det, results_sto


def save_data(results_det, results_sto):
    with open(os.path.join(DATA_DIR, "results_deterministic.json"), "w") as f:
        json.dump(results_det, f, indent=2)
    with open(os.path.join(DATA_DIR, "results_stochastic.json"), "w") as f:
        json.dump(results_sto, f, indent=2)

    with open(os.path.join(DATA_DIR, "summary_deterministic.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "r_step", "category", "pi_time_s", "vi_time_s", "pi_outer_iterations",
            "pi_total_eval_sweeps", "vi_sweeps", "policies_match_pi_vi", "max_V_diff_pi_vs_vi",
        ] + [f"V(s{s})" for s in INITIAL_STATES]
          + [f"steps(s{s})" for s in INITIAL_STATES]
          + [f"outcome(s{s})" for s in INITIAL_STATES])
        for row in results_det:
            writer.writerow(
                [row["r_step"], row["category"], row["pi_time_s"], row["vi_time_s"],
                 row["pi_outer_iterations"], row["pi_total_eval_sweeps"], row["vi_sweeps"],
                 row["policies_match_pi_vi"], row["max_V_diff_pi_vs_vi"]]
                + [row["V_initial_states"][f"s{s}"] for s in INITIAL_STATES]
                + [row["rollouts"][f"s{s}"]["steps"] for s in INITIAL_STATES]
                + [row["rollouts"][f"s{s}"]["outcome"] for s in INITIAL_STATES]
            )

    with open(os.path.join(DATA_DIR, "summary_stochastic.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "r_step", "category", "pi_time_s", "vi_time_s", "pi_outer_iterations",
            "pi_total_eval_sweeps", "vi_sweeps",
        ] + [f"V(s{s})" for s in INITIAL_STATES]
          + [f"success_rate(s{s})" for s in INITIAL_STATES]
          + [f"avg_steps(s{s})" for s in INITIAL_STATES]
          + [f"avg_reward(s{s})" for s in INITIAL_STATES])
        for row in results_sto:
            writer.writerow(
                [row["r_step"], row["category"], row["pi_time_s"], row["vi_time_s"],
                 row["pi_outer_iterations"], row["pi_total_eval_sweeps"], row["vi_sweeps"]]
                + [row["V_initial_states"][f"s{s}"] for s in INITIAL_STATES]
                + [row["rollout_stats"][f"s{s}"]["success_rate"] for s in INITIAL_STATES]
                + [row["rollout_stats"][f"s{s}"]["avg_steps"] for s in INITIAL_STATES]
                + [row["rollout_stats"][f"s{s}"]["avg_reward"] for s in INITIAL_STATES]
            )

    print(f"\nDatos guardados en: {DATA_DIR}")


def _sorted(results):
    return sorted(results, key=lambda r: r["r_step"])


def _shade_good_range(ax):
    ax.axvspan(-2.5, 0.0, color="green", alpha=0.07, label="rango 'bueno' [-2.5, 0]")
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(-2.5, color="gray", linestyle=":", linewidth=1)


def plot_value_vs_rstep(results, title, path):
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in INITIAL_STATES:
        values = [r["V_initial_states"][f"s{s}"] for r in results]
        ax.plot(r_steps, values, marker="o", label=f"V*(s{s})")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("V*(s) del estado inicial")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_iterations_vs_rstep(results, title, path):
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_steps, [r["pi_outer_iterations"] for r in results], marker="o",
            label="PI: iteraciones externas (eval+improve)")
    ax.plot(r_steps, [r["pi_total_eval_sweeps"] for r in results], marker="s",
            label="PI: barridos totales de evaluacion")
    ax.plot(r_steps, [r["vi_sweeps"] for r in results], marker="^",
            label="VI: barridos totales")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("nro de barridos / iteraciones hasta converger")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_time_vs_rstep(results, title, path):
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_steps, [r["pi_time_s"] * 1000 for r in results], marker="o", label="Policy Iteration")
    ax.plot(r_steps, [r["vi_time_s"] * 1000 for r in results], marker="s", label="Value Iteration")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("tiempo de convergencia (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_steps_vs_rstep_deterministic(results, path):
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in INITIAL_STATES:
        steps = [r["rollouts"][f"s{s}"]["steps"] for r in results]
        ax.plot(r_steps, steps, marker="o", label=f"pasos desde s{s}")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("nro de pasos hasta el estado terminal")
    ax.set_title("Pasos hasta terminar el episodio (entorno determinista)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_outcome_vs_rstep_deterministic(results, path):
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    n_goal = []
    for r in results:
        n_goal.append(sum(1 for s in INITIAL_STATES if r["rollouts"][f"s{s}"]["outcome"] == "goal"))
    ax.plot(r_steps, n_goal, marker="o", color="darkgreen")
    ax.set_ylim(-0.3, len(INITIAL_STATES) + 0.3)
    ax.set_yticks(range(len(INITIAL_STATES) + 1))
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel(f"nro de estados iniciales (de {len(INITIAL_STATES)}) que llegan al objetivo")
    ax.set_title("Estados iniciales que alcanzan el objetivo (entorno determinista)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_success_rate_vs_rstep_stochastic(results, path):
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in INITIAL_STATES:
        rates = [r["rollout_stats"][f"s{s}"]["success_rate"] for r in results]
        ax.plot(r_steps, rates, marker="o", label=f"exito desde s{s}")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("tasa de exito (fraccion de episodios que llegan al objetivo)")
    ax.set_title(f"Tasa de exito en {N_STOCHASTIC_EPISODES} episodios simulados (entorno estocastico, prob={STOCHASTIC_PROB})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_avg_reward_vs_rstep_stochastic(results, path):
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in INITIAL_STATES:
        rewards = [r["rollout_stats"][f"s{s}"]["avg_reward"] for r in results]
        ax.plot(r_steps, rewards, marker="o", label=f"retorno medio desde s{s}")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("retorno medio sin descuento")
    ax.set_title(f"Retorno medio en {N_STOCHASTIC_EPISODES} episodios simulados (entorno estocastico)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_rates_vs_rstep_stochastic(results, path):
    """Nuevo: obstacle_rate y timeout_rate (no solo success_rate) vs r_step,
    promediados sobre los estados iniciales. Muestra el 'lado malo' del riesgo,
    no solo el exito."""
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    obstacle = [
        float(np.mean([r["rollout_stats"][f"s{s}"]["obstacle_rate"] for s in INITIAL_STATES]))
        for r in results
    ]
    timeout = [
        float(np.mean([r["rollout_stats"][f"s{s}"]["timeout_rate"] for s in INITIAL_STATES]))
        for r in results
    ]
    success = [
        float(np.mean([r["rollout_stats"][f"s{s}"]["success_rate"] for s in INITIAL_STATES]))
        for r in results
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_steps, success, marker="o", color="darkgreen", label="exito (media)")
    ax.plot(r_steps, obstacle, marker="s", color="crimson", label="obstaculo (media)")
    ax.plot(r_steps, timeout, marker="^", color="darkorange", label="timeout (media)")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("fraccion de episodios (media sobre estados iniciales)")
    ax.set_title(f"Desglose de resultados vs r_step - entorno estocastico ({N_STOCHASTIC_EPISODES} episodios/estado)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reward_variance_vs_rstep_stochastic(results, path):
    """Nuevo: desviacion tipica del retorno (riesgo/variabilidad) vs r_step."""
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in INITIAL_STATES:
        stds = [r["rollout_stats"][f"s{s}"]["std_reward"] for r in results]
        ax.plot(r_steps, stds, marker="o", label=f"std retorno desde s{s}")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("desviacion tipica del retorno")
    ax.set_title(f"Variabilidad del retorno en {N_STOCHASTIC_EPISODES} episodios simulados (entorno estocastico)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy_action_composition_vs_rstep(results, path, title_suffix=""):
    """Nuevo: cuantos de los 20 estados no terminales eligen cada accion
    (arriba-derecha vs abajo-derecha) segun r_step. Muestra como cambia la
    "forma" de la politica optima, no solo su valor."""
    results = _sorted(results)
    r_steps = [r["r_step"] for r in results]
    n_up = []
    n_down = []
    for r in results:
        policy = np.array(r["policy"])[:N_NON_TERMINAL]
        n_up.append(int(np.sum(policy == UP_RIGHT_LOCAL)))
        n_down.append(int(np.sum(policy == DOWN_RIGHT_LOCAL)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_steps, n_up, marker="o", label="nro estados con accion arriba-derecha")
    ax.plot(r_steps, n_down, marker="s", label="nro estados con accion abajo-derecha")
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel(f"nro de estados (de {N_NON_TERMINAL})")
    ax.set_title(f"Composicion de la politica optima vs r_step{title_suffix}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_det_vs_stochastic_value_comparison(results_det, results_sto, path):
    """Nuevo: compara V*(s_init) en el entorno determinista frente al
    estocastico, para ver el 'coste' de la incertidumbre en el valor optimo."""
    results_det = _sorted(results_det)
    results_sto = _sorted(results_sto)
    r_steps = [r["r_step"] for r in results_det]
    fig, axes = plt.subplots(1, len(INITIAL_STATES), figsize=(4.2 * len(INITIAL_STATES), 4.2), sharey=True)
    for ax, s in zip(axes, INITIAL_STATES):
        v_det = [r["V_initial_states"][f"s{s}"] for r in results_det]
        v_sto = [r["V_initial_states"][f"s{s}"] for r in results_sto]
        ax.plot(r_steps, v_det, marker="o", label="determinista")
        ax.plot(r_steps, v_sto, marker="s", label="estocastico")
        _shade_good_range(ax)
        ax.set_xlabel("r_step (r_s)")
        ax.set_title(f"V*(s{s})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("V*(estado inicial)")
    axes[0].legend()
    fig.suptitle("V* determinista vs estocastico por estado inicial", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence_speedup_vs_rstep(results_det, results_sto, path):
    """Nuevo: cuantas veces mas rapido (en tiempo) es VI frente a PI, en
    ambos entornos, para ver si la ventaja de VI depende de r_step."""
    results_det = _sorted(results_det)
    results_sto = _sorted(results_sto)
    r_steps = [r["r_step"] for r in results_det]
    speedup_det = [r["pi_time_s"] / r["vi_time_s"] if r["vi_time_s"] > 0 else np.nan for r in results_det]
    speedup_sto = [r["pi_time_s"] / r["vi_time_s"] if r["vi_time_s"] > 0 else np.nan for r in results_sto]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_steps, speedup_det, marker="o", label="determinista (t_PI / t_VI)")
    ax.plot(r_steps, speedup_sto, marker="s", label="estocastico (t_PI / t_VI)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    _shade_good_range(ax)
    ax.set_xlabel("r_step (r_s)")
    ax.set_ylabel("t_PI / t_VI  (>1 => VI mas rapido)")
    ax.set_title("Ventaja relativa de velocidad: Policy Iteration vs Value Iteration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary_dashboard(results_det, results_sto, path):
    """Nuevo: panel resumen 2x2 con las cuatro metricas mas relevantes juntas,
    para tener una vista rapida de conjunto sin abrir todas las graficas."""
    results_det_s = _sorted(results_det)
    results_sto_s = _sorted(results_sto)
    r_steps = [r["r_step"] for r in results_det_s]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for s in INITIAL_STATES:
        ax.plot(r_steps, [r["V_initial_states"][f"s{s}"] for r in results_det_s], marker="o", label=f"s{s}")
    _shade_good_range(ax)
    ax.set_title("V*(estado inicial) - determinista")
    ax.set_xlabel("r_step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(r_steps, [r["pi_outer_iterations"] for r in results_det_s], marker="o", label="PI iter. ext.")
    ax.plot(r_steps, [r["vi_sweeps"] for r in results_det_s], marker="^", label="VI barridos")
    _shade_good_range(ax)
    ax.set_title("Velocidad de convergencia - determinista")
    ax.set_xlabel("r_step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for s in INITIAL_STATES:
        ax.plot(r_steps, [r["rollout_stats"][f"s{s}"]["success_rate"] for r in results_sto_s], marker="o", label=f"s{s}")
    _shade_good_range(ax)
    ax.set_title(f"Tasa de exito - estocastico (prob={STOCHASTIC_PROB})")
    ax.set_xlabel("r_step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    n_goal = [sum(1 for s in INITIAL_STATES if r["rollouts"][f"s{s}"]["outcome"] == "goal") for r in results_det_s]
    ax.plot(r_steps, n_goal, marker="o", color="darkgreen")
    ax.set_ylim(-0.3, len(INITIAL_STATES) + 0.3)
    _shade_good_range(ax)
    ax.set_title("Estados iniciales que alcanzan el objetivo - determinista")
    ax.set_xlabel("r_step")
    ax.grid(alpha=0.3)

    fig.suptitle("Panel resumen: sensibilidad a r_step", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


UP_RIGHT_LOCAL = 0
DOWN_RIGHT_LOCAL = 1


def make_all_plots(results_det, results_sto):
    plot_value_vs_rstep(
        results_det, "V*(estado inicial) vs r_step - entorno determinista",
        os.path.join(PLOTS_DIR, "value_vs_rstep_deterministic.png"),
    )
    plot_value_vs_rstep(
        results_sto, "V*(estado inicial) vs r_step - entorno estocastico",
        os.path.join(PLOTS_DIR, "value_vs_rstep_stochastic.png"),
    )
    plot_iterations_vs_rstep(
        results_det, "Iteraciones hasta converger vs r_step - entorno determinista",
        os.path.join(PLOTS_DIR, "iterations_vs_rstep_deterministic.png"),
    )
    plot_iterations_vs_rstep(
        results_sto, "Iteraciones hasta converger vs r_step - entorno estocastico",
        os.path.join(PLOTS_DIR, "iterations_vs_rstep_stochastic.png"),
    )
    plot_time_vs_rstep(
        results_det, "Tiempo de convergencia vs r_step - entorno determinista",
        os.path.join(PLOTS_DIR, "time_vs_rstep_deterministic.png"),
    )
    plot_steps_vs_rstep_deterministic(
        results_det, os.path.join(PLOTS_DIR, "steps_vs_rstep_deterministic.png")
    )
    plot_outcome_vs_rstep_deterministic(
        results_det, os.path.join(PLOTS_DIR, "outcomes_vs_rstep_deterministic.png")
    )
    plot_success_rate_vs_rstep_stochastic(
        results_sto, os.path.join(PLOTS_DIR, "success_rate_vs_rstep_stochastic.png")
    )
    plot_avg_reward_vs_rstep_stochastic(
        results_sto, os.path.join(PLOTS_DIR, "avg_reward_vs_rstep_stochastic.png")
    )
    save_policy_grid_overview(
        _sorted(results_det), os.path.join(PLOTS_DIR, "policy_grids_overview.png")
    )

    # ---------------------- graficas nuevas (adicionales) ----------------------
    plot_risk_rates_vs_rstep_stochastic(
        results_sto, os.path.join(PLOTS_DIR, "risk_rates_vs_rstep_stochastic.png")
    )
    plot_reward_variance_vs_rstep_stochastic(
        results_sto, os.path.join(PLOTS_DIR, "reward_variance_vs_rstep_stochastic.png")
    )
    plot_policy_action_composition_vs_rstep(
        results_det, os.path.join(PLOTS_DIR, "policy_composition_vs_rstep_deterministic.png"),
        title_suffix=" (entorno determinista)",
    )
    plot_policy_action_composition_vs_rstep(
        results_sto, os.path.join(PLOTS_DIR, "policy_composition_vs_rstep_stochastic.png"),
        title_suffix=" (entorno estocastico)",
    )
    plot_det_vs_stochastic_value_comparison(
        results_det, results_sto, os.path.join(PLOTS_DIR, "value_det_vs_stochastic_comparison.png")
    )
    plot_convergence_speedup_vs_rstep(
        results_det, results_sto, os.path.join(PLOTS_DIR, "convergence_speedup_vs_rstep.png")
    )
    plot_summary_dashboard(
        results_det, results_sto, os.path.join(PLOTS_DIR, "summary_dashboard.png")
    )

    print(f"Graficas guardadas en: {PLOTS_DIR}")


if __name__ == "__main__":
    print("Ejecutando experimentos de sensibilidad a r_step...")
    print(f"Valores de r_step: {[c['value'] for c in R_STEP_EXPERIMENTS]}")
    print(f"GAMMA={GAMMA} | tol={CONVERGENCE_TOLERANCE} | prob_estocastico={STOCHASTIC_PROB} "
          f"| max_pasos_rollout={MAX_ROLLOUT_STEPS} | episodios_estocasticos={N_STOCHASTIC_EPISODES}")

    results_det, results_sto = run_all_experiments()
    save_data(results_det, results_sto)
    make_all_plots(results_det, results_sto)

    print("\nListo. Revisa la carpeta 'results_2/' (datos en 'data/', graficas en 'plots/').")