"""Experimentos: influencia de GAMMA y de CONVERGENCE_TOLERANCE sobre Policy
Iteration y Value Iteration en el entorno "Jump To The Goal".

Se calcula UNA malla completa gamma x tol (para entorno determinista y
estocastico) y de ella se sacan:
  * barrido de gamma  (tol fija = BASE_TOL)
  * barrido de tol    (gamma fija = BASE_GAMMA)
  * mapas de calor conjuntos gamma x tol
r_step, r_obstacle y r_goal se mantienen en los valores por defecto del enunciado.
"""

import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
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
from policy_iteration import policy_evaluation, policy_improvement, policy_iteration
from value_iteration import value_iteration

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
EXACT_TOL = 1e-13  # "solucion exacta" de referencia
ERR_FLOOR = 1e-16  # para poder dibujar ceros en escala log
UP, DOWN = 0, 1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_3")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
for d in (DATA_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

KINDS = ("det", "sto")
KIND_LABEL = {"det": "determinista", "sto": f"estocastico (prob={STOCHASTIC_PROB})"}
KIND_COLOR = {"det": "tab:blue", "sto": "tab:orange"}


# ============================================================================
# Versiones instrumentadas (mismo algoritmo que el vuestro, pero devuelven
# contadores y el historial de delta)
# ============================================================================
def vi_instrumented(P, nS, nA, gamma, tol, max_sweeps=100_000):
    V = np.zeros(nS)
    deltas = []
    while len(deltas) < max_sweeps:
        delta = 0.0
        for s in range(nS):
            old = V[s]
            V[s] = max(
                sum(p * (r + gamma * V[n]) for p, n, r, _ in P[s][a]) for a in range(nA)
            )
            delta = max(delta, abs(old - V[s]))
        deltas.append(delta)
        if delta < tol:
            break
    policy = np.array(
        [
            max(range(nA), key=lambda a: sum(p * (r + gamma * V[n]) for p, n, r, _ in P[s][a]))
            for s in range(nS)
        ]
    )
    return V, policy, len(deltas), deltas


def pe_instrumented(P, nS, nA, policy, gamma, tol):
    V = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for s in range(nS):
            old = V[s]
            V[s] = sum(p * (r + gamma * V[n]) for p, n, r, _ in P[s][int(policy[s])])
            delta = max(delta, abs(old - V[s]))
        if delta < tol:
            break
    return V, sweeps


def pi_instrumented(P, nS, nA, gamma, tol):
    policy = np.zeros(nS, dtype=int)
    outer, eval_sweeps = 0, 0
    while True:
        outer += 1
        V, sw = pe_instrumented(P, nS, nA, policy, gamma, tol)
        eval_sweeps += sw
        new = policy_improvement(P, nS, nA, V, gamma)
        done = np.array_equal(policy, new)
        policy = new
        if done:
            break
    return V, policy, outer, eval_sweeps


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
    V_pi = policy_evaluation(P, nS, nA, policy, gamma, EXACT_TOL)
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
# Ejecucion de la malla completa
# ============================================================================
def run_grid():
    envs = {
        "det": JumpToTheGoalEnv(deterministic=True, prob=1.0, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
        "sto": JumpToTheGoalEnv(deterministic=False, prob=STOCHASTIC_PROB, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
    }
    rows, delta_hist, opt_sets = [], {k: {} for k in KINDS}, {k: {} for k in KINDS}

    for kind in KINDS:
        env = envs[kind]
        P, nS, nA = env.P, env.nS, env.nA
        print(f"\n##### entorno {KIND_LABEL[kind]} #####")

        for gamma in GAMMAS:
            # --- solucion exacta de referencia para este gamma
            V_ref, _, _, ref_deltas = vi_instrumented(P, nS, nA, gamma, EXACT_TOL)
            if ref_deltas[-1] > 0:
                print(f"  [aviso] gamma={gamma}: la referencia no llego a delta=0")
            delta_hist[kind][gamma] = ref_deltas
            opt_sets[kind][gamma] = optimal_action_sets(P, nS, nA, V_ref, gamma)
            print(f"  gamma={gamma}: VI llega a delta=0 tras {len(ref_deltas)} barridos")

            for tol in TOLS:
                V_vi, pol_vi, vi_sw, _ = vi_instrumented(P, nS, nA, gamma, tol)
                V_pi, pol_pi, pi_out, pi_ev = pi_instrumented(P, nS, nA, gamma, tol)

                row = {
                    "env": kind, "gamma": gamma, "tol": tol,
                    "vi_sweeps": vi_sw,
                    "pi_outer": pi_out,
                    "pi_eval_sweeps": pi_ev,
                    "vi_time_ms": best_time_ms(value_iteration, P, nS, nA, gamma, tol),
                    "pi_time_ms": best_time_ms(policy_iteration, P, nS, nA, gamma, tol),
                    "err_V_vi": float(np.max(np.abs(V_vi - V_ref))),
                    "err_V_pi": float(np.max(np.abs(V_pi - V_ref))),
                    "regret_vi": policy_regret(P, nS, nA, pol_vi, V_ref, gamma),
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

    # --- iteraciones extra respecto a la tolerancia base (mismo gamma)
    base = {(r["env"], r["gamma"]): r for r in rows if r["tol"] == BASE_TOL}
    for r in rows:
        b = base[(r["env"], r["gamma"])]
        r["extra_vi_sweeps"] = r["vi_sweeps"] - b["vi_sweeps"]
        r["extra_pi_eval_sweeps"] = r["pi_eval_sweeps"] - b["pi_eval_sweeps"]
        r["extra_pi_outer"] = r["pi_outer"] - b["pi_outer"]

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
# Guardado de datos
# ============================================================================
def save_data(rows, delta_hist):
    with open(os.path.join(DATA_DIR, "grid_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(DATA_DIR, "vi_delta_history.json"), "w") as f:
        json.dump({k: {str(g): v for g, v in d.items()} for k, d in delta_hist.items()}, f)

    cols = ["env", "gamma", "tol", "vi_sweeps", "pi_outer", "pi_eval_sweeps",
            "extra_vi_sweeps", "extra_pi_outer", "extra_pi_eval_sweeps",
            "vi_time_ms", "pi_time_ms", "err_V_vi", "err_V_pi", "regret_vi", "regret_pi"]
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
# BLOQUE 1: barrido de gamma (tol = BASE_TOL)
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
    axes[0].set_ylabel("V*(estado inicial)")
    axes[0].legend()
    fig.suptitle(f"V*(s_init) vs gamma (tol={BASE_TOL:g})")
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
    ax.set_title("Cambios en la politica optima")
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
    ax.set_title("Composicion de la politica")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, name)


def plot_gamma_convergence(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = gamma_slice(rows, kind)
        ax.plot(GAMMAS, [r["vi_sweeps"] for r in sl], marker="^", label="VI: barridos")
        ax.plot(GAMMAS, [r["pi_eval_sweeps"] for r in sl], marker="s", label="PI: barridos de evaluacion")
        ax.plot(GAMMAS, [r["pi_outer"] for r in sl], marker="o", label="PI: iteraciones externas")
        _mark_base_gamma(ax)
        ax.set_title(KIND_LABEL[kind])
        ax.set_xlabel("gamma")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("barridos / iteraciones hasta converger")
    axes[0].legend()
    fig.suptitle(f"Coste de convergencia vs gamma (tol={BASE_TOL:g})")
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
    fig.suptitle("Politica optima determinista - comparacion por gamma", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOTS_DIR, name), dpi=140)
    plt.close(fig)


# ============================================================================
# BLOQUE 2: barrido de tolerancia (gamma = BASE_GAMMA)
# ============================================================================
def plot_tol_sweeps(rows, name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j, kind in enumerate(KINDS):
        sl = tol_slice(rows, kind)
        ax = axes[0, j]
        ax.plot(TOLS, [r["vi_sweeps"] for r in sl], marker="^", label="VI: barridos")
        ax.plot(TOLS, [r["pi_eval_sweeps"] for r in sl], marker="s", label="PI: barridos de evaluacion")
        ax.plot(TOLS, [r["pi_outer"] for r in sl], marker="o", label="PI: iteraciones externas")
        ax.set_xscale("log")
        _mark_base_tol(ax)
        ax.set_title(f"{KIND_LABEL[kind]} (gamma={BASE_GAMMA})")
        ax.set_ylabel("barridos / iteraciones hasta converger")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend()

        ax = axes[1, j]
        ax.plot(TOLS, [r["extra_vi_sweeps"] for r in sl], marker="^", label="VI")
        ax.plot(TOLS, [r["extra_pi_eval_sweeps"] for r in sl], marker="s", label="PI (evaluacion)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        _mark_base_tol(ax)
        ax.set_xlabel("tolerancia")
        ax.set_ylabel(f"barridos extra vs tol={BASE_TOL:g}")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend()
    fig.suptitle("Coste de convergencia vs tolerancia (y barridos extra sobre la tolerancia base)")
    _save(fig, name)


def plot_tol_accuracy(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    bound = [BASE_GAMMA * t / (1 - BASE_GAMMA) for t in TOLS]

    ax = axes[0]
    for kind in KINDS:
        sl = tol_slice(rows, kind)
        ax.plot(TOLS, [max(r["err_V_vi"], ERR_FLOOR) for r in sl], marker="^",
                color=KIND_COLOR[kind], label=f"VI {kind}")
        ax.plot(TOLS, [max(r["err_V_pi"], ERR_FLOOR) for r in sl], marker="s", linestyle="--",
                color=KIND_COLOR[kind], label=f"PI {kind}")
    ax.plot(TOLS, bound, color="gray", linestyle=":", label="cota gamma*tol/(1-gamma)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _mark_base_tol(ax)
    ax.set_xlabel("tolerancia")
    ax.set_ylabel(f"max |V - V_exacta|   (los ceros se dibujan en {ERR_FLOOR:g})")
    ax.set_title("Error en la funcion de valor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for kind in KINDS:
        sl = tol_slice(rows, kind)
        ax.plot(TOLS, [max(r["regret_vi"], ERR_FLOOR) for r in sl], marker="^",
                color=KIND_COLOR[kind], label=f"VI {kind}")
        ax.plot(TOLS, [max(r["regret_pi"], ERR_FLOOR) for r in sl], marker="s", linestyle="--",
                color=KIND_COLOR[kind], label=f"PI {kind}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _mark_base_tol(ax)
    ax.set_xlabel("tolerancia")
    ax.set_ylabel(f"regret maximo de la politica  (0 -> {ERR_FLOOR:g})")
    ax.set_title("Calidad de la politica devuelta")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
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
            if (~pos).any():  # delta exactamente 0: convergencia exacta
                ax.plot(k[~pos][:1], [1e-14], marker="X", color=line.get_color(), markersize=9)
        for t in (1.0, 1e-3, 1e-6):
            ax.axhline(t, color="gray", linestyle=":", linewidth=1)
            ax.text(ax.get_xlim()[1], t, f" tol={t:g}", va="bottom", ha="right", fontsize=7, color="gray")
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 50)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("barrido de Value Iteration")
        ax.set_title(KIND_LABEL[kind])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("delta (maximo cambio en el barrido)")
    axes[0].legend()
    fig.suptitle("Curva de residuo de VI: la tolerancia decide donde se corta (X = delta exactamente 0)")
    _save(fig, name)


# ============================================================================
# BLOQUE 3: mapas de calor gamma x tol
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
    ax.set_xlabel("tolerancia")
    ax.set_ylabel("gamma")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def plot_heatmaps_sweeps(rows, name):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _heatmap(axes[0, 0], rows, "det", "vi_sweeps", "VI barridos - determinista")
    _heatmap(axes[0, 1], rows, "sto", "vi_sweeps", "VI barridos - estocastico")
    _heatmap(axes[1, 0], rows, "det", "pi_eval_sweeps", "PI barridos de evaluacion totales - determinista")
    _heatmap(axes[1, 1], rows, "sto", "pi_eval_sweeps", "PI barridos de evaluacion totales - estocastico")
    fig.suptitle("Coste de convergencia en la malla gamma x tolerancia")
    _save(fig, name)


def plot_heatmaps_accuracy(rows, name):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _heatmap(axes[0, 0], rows, "det", "err_V_vi", "log10 max|V - V_exacta| (VI) - determinista",
             log_floor=True, cmap="magma")
    _heatmap(axes[0, 1], rows, "sto", "err_V_vi", "log10 max|V - V_exacta| (VI) - estocastico",
             log_floor=True, cmap="magma")
    _heatmap(axes[1, 0], rows, "det", "regret_vi", "log10 regret de la politica (VI) - determinista",
             log_floor=True, cmap="magma")
    _heatmap(axes[1, 1], rows, "sto", "regret_vi", "log10 regret de la politica (VI) - estocastico",
             log_floor=True, cmap="magma")
    fig.suptitle(f"Precision en la malla gamma x tolerancia (0 = exacto; piso en 1e{int(np.log10(ERR_FLOOR))})")
    _save(fig, name)


def make_all_plots(rows, delta_hist, opt_sets):
    # gamma
    plot_gamma_value(rows, "gamma_value.png")
    plot_gamma_policy_change(rows, opt_sets, "gamma_policy_change.png")
    plot_gamma_convergence(rows, "gamma_convergence.png")
    plot_gamma_outcomes(rows, "gamma_outcomes.png")
    plot_gamma_policy_grids(rows, "gamma_policy_grids.png")
    # tolerancia
    plot_tol_sweeps(rows, "tol_sweeps.png")
    plot_tol_accuracy(rows, "tol_accuracy.png")
    plot_delta_curves(delta_hist, "tol_delta_curves.png")
    # conjunto
    plot_heatmaps_sweeps(rows, "joint_heatmaps_sweeps.png")
    plot_heatmaps_accuracy(rows, "joint_heatmaps_accuracy.png")
    print(f"Graficas guardadas en: {PLOTS_DIR}")


if __name__ == "__main__":
    print("Ejecutando barrido de gamma y tolerancia...")
    print(f"GAMMAS={GAMMAS}\nTOLS={TOLS}\nbase: gamma={BASE_GAMMA}, tol={BASE_TOL:g} | "
          f"r_step={R_STEP}, r_obs={R_OBSTACLE}, r_goal={R_GOAL}")
    rows, delta_hist, opt_sets = run_grid()
    save_data(rows, delta_hist)
    make_all_plots(rows, delta_hist, opt_sets)
    print("\nListo. Revisa 'results_3/' (datos en 'data/', graficas en 'plots/').")