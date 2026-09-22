"""Experimento 2 · Comparacion Value Iteration vs Policy Iteration.

======================================================================
PREGUNTA QUE RESPONDE ESTE SCRIPT
----------------------------------------------------------------------
"Comparar el desempeño del agente cuando aprende mediante Policy
Iteration y cuando lo hace con Value Iteration. ¿Se llega siempre a la
misma solucion?, ¿son las politicas optimas iguales?, ¿convergen en
tiempos similares?"
======================================================================

A diferencia de `experiment_pi_hyperparams.py` (que solo usa PI para
analizar el efecto de gamma/tol), aqui SI se ejecutan los dos
algoritmos, con los MISMOS (gamma, tol, entorno) en cada punto, para
poder compararlos de forma justa.

Que se mide, para el entorno determinista y el estocastico:

  BLOQUE 1 - Misma solucion?
    - max |V_PI - V_VI| en cada (gamma, tol) -> si es ~0, ambos
      algoritmos convergen al mismo V*.

  BLOQUE 2 - Misma politica optima?
    - Numero de estados no terminales en los que pi_PI(s) != pi_VI(s).
    - Se distingue entre diferencias "reales" (una de las dos acciones
      NO es optima) y diferencias por EMPATE (ambas acciones son
      optimas con Q(s,a) casi iguales; el desempate depende solo de
      como recorre argmax cada implementacion, no de que una politica
      sea peor que la otra).

  BLOQUE 3 - Convergen en tiempos similares?
    - Tiempo de convergencia (ms) de cada algoritmo vs gamma y vs tol.
    - Coste computacional "normalizado": numero total de backups de
      Bellman (evaluaciones de Q(s,a)) que hace cada algoritmo, que es
      una medida independiente de la maquina donde se ejecuta.
    - Mapas de calor gamma x tol del ratio de tiempo y de backups
      (PI / VI) en ambos entornos.

  BLOQUE 4 - Verificacion visual
    - Rejillas de politica optima de PI y de VI lado a lado para un
      caso representativo, marcando en rojo los estados en los que
      discrepan (si los hay).

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
from value_iteration import value_iteration

# ============================================================================
# Configuracion
# ============================================================================
BASE_GAMMA = 0.9
BASE_TOL = 1e-3

GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
TOLS = [10.0, 3.0, 1.0, 0.3, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

STOCHASTIC_PROB = 0.8
TIMING_REPEATS = 7
EXACT_TOL = 1e-13
TIE_EPS = 1e-6  # margen para considerar dos Q(s,a) "empatadas"
UP, DOWN = 0, 1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_vi_vs_pi")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
for d in (DATA_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

KINDS = ("det", "sto")
KIND_LABEL = {"det": "determinista", "sto": f"estocastico (prob={STOCHASTIC_PROB})"}
KIND_COLOR = {"det": "tab:blue", "sto": "tab:orange"}
ALGO_COLOR = {"PI": "tab:purple", "VI": "tab:green"}


# ============================================================================
# Versiones instrumentadas: mismos algoritmos, pero cuentan barridos/backups
# ============================================================================
def policy_evaluation_instrumented(P, nS, nA, policy, gamma, tol):
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
        V, sw = policy_evaluation_instrumented(P, nS, nA, policy, gamma, tol)
        eval_sweeps += sw
        new_policy = policy_improvement(P, nS, nA, V, gamma)
        done = np.array_equal(policy, new_policy)
        policy = new_policy
        if done:
            break
    return V, policy, outer, eval_sweeps


def vi_instrumented(P, nS, nA, gamma, tol):
    V = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for s in range(nS):
            old = V[s]
            V[s] = max(
                sum(p * (r + gamma * V[n]) for p, n, r, _ in P[s][a]) for a in range(nA)
            )
            delta = max(delta, abs(old - V[s]))
        if delta < tol:
            break
    policy = np.array(
        [
            max(range(nA), key=lambda a: sum(p * (r + gamma * V[n]) for p, n, r, _ in P[s][a]))
            for s in range(nS)
        ]
    )
    return V, policy, sweeps


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


def compare_policies(P, nS, nA, pol_a, pol_b, V_ref, gamma, eps=TIE_EPS):
    """Compara dos politicas estado a estado sobre los estados no terminales.

    Devuelve (n_diff_total, n_diff_reales, n_diff_por_empate), usando Q
    calculada sobre una V de referencia de alta precision para decidir si un
    estado en el que difieren es un empate legitimo (ambas acciones son
    ~optimas) o una discrepancia real (una de las dos no lo es).
    """
    Q = q_values(P, nS, nA, V_ref, gamma)
    n_diff = 0
    n_tie = 0
    n_real = 0
    for s in range(nS - 1):  # sin el estado terminal
        if pol_a[s] == pol_b[s]:
            continue
        n_diff += 1
        best = Q[s].max()
        both_optimal = (Q[s][pol_a[s]] >= best - eps) and (Q[s][pol_b[s]] >= best - eps)
        if both_optimal:
            n_tie += 1
        else:
            n_real += 1
    return n_diff, n_real, n_tie


# ============================================================================
# Ejecucion de la malla completa gamma x tol, para PI y VI a la vez
# ============================================================================
def run_grid():
    envs = {
        "det": JumpToTheGoalEnv(deterministic=True, prob=1.0, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
        "sto": JumpToTheGoalEnv(deterministic=False, prob=STOCHASTIC_PROB, r_step=R_STEP,
                                r_obstacle=R_OBSTACLE, r_goal=R_GOAL),
    }
    rows = []

    for kind in KINDS:
        env = envs[kind]
        P, nS, nA = env.P, env.nS, env.nA
        print(f"\n##### entorno {KIND_LABEL[kind]} #####")

        # referencia de muy alta precision (para decidir empates de forma robusta)
        V_ref, _, _, _ = pi_instrumented(P, nS, nA, BASE_GAMMA, EXACT_TOL)

        for gamma in GAMMAS:
            V_ref_g, _, _, _ = pi_instrumented(P, nS, nA, gamma, EXACT_TOL)

            for tol in TOLS:
                V_pi, pol_pi, pi_outer, pi_eval_sweeps = pi_instrumented(P, nS, nA, gamma, tol)
                V_vi, pol_vi, vi_sweeps = vi_instrumented(P, nS, nA, gamma, tol)

                pi_time_ms = best_time_ms(policy_iteration, P, nS, nA, gamma, tol)
                vi_time_ms = best_time_ms(value_iteration, P, nS, nA, gamma, tol)

                # coste computacional independiente de la maquina: nro de
                # veces que se evalua un Q(s,a) = suma sobre transiciones
                pi_backups = pi_eval_sweeps * nS + pi_outer * nS * nA
                vi_backups = vi_sweeps * nS * nA

                n_diff, n_real, n_tie = compare_policies(P, nS, nA, pol_pi, pol_vi, V_ref_g, gamma)

                rows.append({
                    "env": kind, "gamma": gamma, "tol": tol,
                    "max_V_diff": float(np.max(np.abs(V_pi - V_vi))),
                    "policies_equal": bool(n_diff == 0),
                    "n_diff_states": n_diff,
                    "n_diff_real": n_real,
                    "n_diff_tie": n_tie,
                    "pi_time_ms": pi_time_ms,
                    "vi_time_ms": vi_time_ms,
                    "time_ratio_pi_over_vi": pi_time_ms / vi_time_ms if vi_time_ms > 0 else np.nan,
                    "pi_outer": pi_outer,
                    "pi_eval_sweeps": pi_eval_sweeps,
                    "vi_sweeps": vi_sweeps,
                    "pi_backups": int(pi_backups),
                    "vi_backups": int(vi_backups),
                    "backups_ratio_pi_over_vi": pi_backups / vi_backups if vi_backups > 0 else np.nan,
                    "V_pi": V_pi.tolist(),
                    "V_vi": V_vi.tolist(),
                    "policy_pi": pol_pi.tolist(),
                    "policy_vi": pol_vi.tolist(),
                })
            print(f"  gamma={gamma}: listo (barrido completo de tol)")

    return rows


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
def save_data(rows):
    with open(os.path.join(DATA_DIR, "grid_results.json"), "w") as f:
        json.dump(rows, f, indent=1)

    cols = ["env", "gamma", "tol", "max_V_diff", "policies_equal", "n_diff_states",
            "n_diff_real", "n_diff_tie", "pi_time_ms", "vi_time_ms",
            "time_ratio_pi_over_vi", "pi_outer", "pi_eval_sweeps", "vi_sweeps",
            "pi_backups", "vi_backups", "backups_ratio_pi_over_vi"]
    with open(os.path.join(DATA_DIR, "grid_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"\nDatos guardados en: {DATA_DIR}")


def print_headline_summary(rows):
    """Resumen textual rapido para pegar directamente en la memoria."""
    print("\n" + "=" * 78)
    print("RESUMEN: VI vs PI")
    print("=" * 78)
    for kind in KINDS:
        sl = gamma_slice(rows, kind)
        max_diff = max(r["max_V_diff"] for r in sl)
        n_mismatch = sum(1 for r in sl if not r["policies_equal"])
        n_real_mismatch = sum(1 for r in sl if r["n_diff_real"] > 0)
        avg_time_ratio = np.mean([r["time_ratio_pi_over_vi"] for r in sl])
        avg_backup_ratio = np.mean([r["backups_ratio_pi_over_vi"] for r in sl])
        print(f"\n[{KIND_LABEL[kind]}] (barrido de gamma, tol={BASE_TOL:g})")
        print(f"  max|V_PI - V_VI| en todo el barrido de gamma : {max_diff:.2e}")
        print(f"  gammas con politicas PI/VI distintas          : {n_mismatch}/{len(GAMMAS)} "
              f"(de ellos, con discrepancia REAL no explicada por empates: {n_real_mismatch})")
        print(f"  tiempo medio PI/VI (ratio)                    : {avg_time_ratio:.2f}x")
        print(f"  backups Bellman medios PI/VI (ratio)          : {avg_backup_ratio:.2f}x")
    print("=" * 78)


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
# BLOQUE 1: misma solucion? (V*)
# ============================================================================
def plot_value_overlay(rows, name):
    """V*(s_init) de PI y de VI superpuestos: si coinciden, las lineas se
    solapan por completo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = gamma_slice(rows, kind)
        for s in INITIAL_STATES:
            v_pi = [np.array(r["V_pi"])[s] for r in sl]
            v_vi = [np.array(r["V_vi"])[s] for r in sl]
            ax.plot(GAMMAS, v_pi, marker="o", linewidth=2.5, alpha=0.6, label=f"PI s{s}")
            ax.plot(GAMMAS, v_vi, marker="x", linewidth=1, linestyle="--", label=f"VI s{s}")
        _mark_base_gamma(ax)
        ax.set_title(KIND_LABEL[kind])
        ax.set_xlabel("gamma")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("V*(estado inicial)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"[PI vs VI] V*(s_init): lineas solapadas => misma solucion (tol={BASE_TOL:g})")
    _save(fig, name)


def plot_value_diff(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl_g = gamma_slice(rows, kind)
        sl_t = tol_slice(rows, kind)
        ax.plot(GAMMAS, [max(r["max_V_diff"], 1e-16) for r in sl_g], marker="o", label="vs gamma (tol fija)")
        ax2 = ax.twiny()
        ax2.plot(TOLS, [max(r["max_V_diff"], 1e-16) for r in sl_t], marker="s", color="tab:red",
                 label="vs tol (gamma fija)")
        ax2.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("gamma (linea azul)")
        ax2.set_xlabel("tolerancia (linea roja)")
        ax.set_title(KIND_LABEL[kind])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("max |V_PI - V_VI|")
    fig.suptitle("[PI vs VI] Diferencia entre las soluciones de V* (deberia ser ~0)")
    _save(fig, name)


# ============================================================================
# BLOQUE 2: misma politica optima?
# ============================================================================
def plot_policy_agreement(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, kind in zip(axes, KINDS):
        sl = gamma_slice(rows, kind)
        ax.plot(GAMMAS, [r["n_diff_states"] for r in sl], marker="o", color="black",
                label="estados donde pi_PI != pi_VI (total)")
        ax.plot(GAMMAS, [r["n_diff_tie"] for r in sl], marker="s", color="tab:blue",
                label="...de ellos, por EMPATE (ambas acciones optimas)")
        ax.plot(GAMMAS, [r["n_diff_real"] for r in sl], marker="^", color="tab:red",
                label="...de ellos, discrepancia REAL")
        _mark_base_gamma(ax)
        ax.set_ylim(-0.3, N_NON_TERMINAL * 0.3 + 1)
        ax.set_xlabel("gamma")
        ax.set_title(KIND_LABEL[kind])
        ax.grid(alpha=0.3)
        if kind == "det":
            ax.legend(fontsize=8)
    axes[0].set_ylabel(f"nro de estados (de {N_NON_TERMINAL})")
    fig.suptitle(f"[PI vs VI] Coincidencia de la politica optima vs gamma (tol={BASE_TOL:g})")
    _save(fig, name)


def plot_policy_agreement_vs_tol(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = tol_slice(rows, kind)
        ax.plot(TOLS, [r["n_diff_states"] for r in sl], marker="o", color="black",
                label="total")
        ax.plot(TOLS, [r["n_diff_tie"] for r in sl], marker="s", color="tab:blue", label="empate")
        ax.plot(TOLS, [r["n_diff_real"] for r in sl], marker="^", color="tab:red", label="real")
        ax.set_xscale("log")
        _mark_base_tol(ax)
        ax.set_xlabel("tolerancia (theta)")
        ax.set_title(f"{KIND_LABEL[kind]} (gamma={BASE_GAMMA})")
        ax.grid(alpha=0.3)
        if kind == "det":
            ax.legend(fontsize=8)
    axes[0].set_ylabel(f"estados con politica distinta (de {N_NON_TERMINAL})")
    fig.suptitle("[PI vs VI] Coincidencia de la politica optima vs tolerancia")
    _save(fig, name)


def plot_policy_grids_side_by_side(rows, kind, gamma, tol, name):
    """Rejillas de politica de PI y VI lado a lado, marcando en rojo los
    estados donde discrepan."""
    r = get(rows, kind, gamma, tol)
    V_pi, pol_pi = np.array(r["V_pi"]), np.array(r["policy_pi"])
    V_vi, pol_vi = np.array(r["V_vi"]), np.array(r["policy_vi"])
    nrow, ncol = len(GRID_WORLD), len(GRID_WORLD[0])
    arrow = {UP: (0.32, 0.32), DOWN: (0.32, -0.32)}

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    for ax, V, pol, title in ((axes[0], V_pi, pol_pi, "Policy Iteration"),
                               (axes[1], V_vi, pol_vi, "Value Iteration")):
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
                    mismatch = pol_pi[st] != pol_vi[st]
                    dx, dy = arrow[int(pol[st])]
                    color = "red" if mismatch else "black"
                    ax.annotate("", xy=(col_i + dx, row_i - dy),
                                xytext=(col_i - dx * 0.2, row_i + dy * 0.2),
                                arrowprops=dict(arrowstyle="->", color=color, lw=2.0 if mismatch else 1.4))
                    ax.text(col_i, row_i + 0.32, f"s{st}\n{V[st]:.1f}", ha="center",
                            va="center", fontsize=6,
                            color="red" if mismatch else "black",
                            fontweight="bold" if mismatch else "normal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
    n_diff = r["n_diff_states"]
    fig.suptitle(f"[PI vs VI] gamma={gamma}, tol={tol:g}, entorno {KIND_LABEL[kind]}\n"
                 f"estados con politica distinta: {n_diff} (en rojo)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(PLOTS_DIR, name), dpi=150)
    plt.close(fig)


# ============================================================================
# BLOQUE 3: convergen en tiempos similares?
# ============================================================================
def plot_time_comparison(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = gamma_slice(rows, kind)
        ax.plot(GAMMAS, [r["pi_time_ms"] for r in sl], marker="o", color=ALGO_COLOR["PI"], label="Policy Iteration")
        ax.plot(GAMMAS, [r["vi_time_ms"] for r in sl], marker="s", color=ALGO_COLOR["VI"], label="Value Iteration")
        _mark_base_gamma(ax)
        ax.set_xlabel("gamma")
        ax.set_title(KIND_LABEL[kind])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("tiempo de convergencia (ms)")
    axes[0].legend()
    fig.suptitle(f"[PI vs VI] Tiempo de convergencia vs gamma (tol={BASE_TOL:g})")
    _save(fig, name)


def plot_time_comparison_vs_tol(rows, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = tol_slice(rows, kind)
        ax.plot(TOLS, [r["pi_time_ms"] for r in sl], marker="o", color=ALGO_COLOR["PI"], label="Policy Iteration")
        ax.plot(TOLS, [r["vi_time_ms"] for r in sl], marker="s", color=ALGO_COLOR["VI"], label="Value Iteration")
        ax.set_xscale("log")
        _mark_base_tol(ax)
        ax.set_xlabel("tolerancia (theta)")
        ax.set_title(f"{KIND_LABEL[kind]} (gamma={BASE_GAMMA})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("tiempo de convergencia (ms)")
    axes[0].legend()
    fig.suptitle("[PI vs VI] Tiempo de convergencia vs tolerancia")
    _save(fig, name)


def plot_backups_comparison(rows, name):
    """Coste computacional independiente de la maquina (nro de veces que se
    evalua un Q(s,a)), para poder comparar sin depender de la velocidad del
    hardware donde se ejecuten los scripts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sl = gamma_slice(rows, kind)
        ax.plot(GAMMAS, [r["pi_backups"] for r in sl], marker="o", color=ALGO_COLOR["PI"], label="Policy Iteration")
        ax.plot(GAMMAS, [r["vi_backups"] for r in sl], marker="s", color=ALGO_COLOR["VI"], label="Value Iteration")
        _mark_base_gamma(ax)
        ax.set_yscale("log")
        ax.set_xlabel("gamma")
        ax.set_title(KIND_LABEL[kind])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("backups de Bellman hasta converger (escala log)")
    axes[0].legend()
    fig.suptitle(f"[PI vs VI] Coste computacional (independiente de la maquina) vs gamma (tol={BASE_TOL:g})")
    _save(fig, name)


def plot_heatmaps_ratio(rows, name):
    def _heat(ax, kind, key, title, cmap="coolwarm", center=1.0):
        Z = np.array([[get(rows, kind, g, t)[key] for t in TOLS] for g in GAMMAS], dtype=float)
        vmax = np.nanmax(np.abs(np.log2(Z)))
        im = ax.imshow(np.log2(Z), cmap=cmap, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(TOLS)))
        ax.set_xticklabels([f"{t:g}" for t in TOLS], rotation=45, fontsize=7)
        ax.set_yticks(range(len(GAMMAS)))
        ax.set_yticklabels([f"{g:g}" for g in GAMMAS], fontsize=8)
        for i in range(len(GAMMAS)):
            for j in range(len(TOLS)):
                ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=6.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tolerancia (theta)")
        ax.set_ylabel("gamma")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log2(ratio)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _heat(axes[0, 0], "det", "time_ratio_pi_over_vi", "ratio tiempo PI/VI - determinista")
    _heat(axes[0, 1], "sto", "time_ratio_pi_over_vi", "ratio tiempo PI/VI - estocastico")
    _heat(axes[1, 0], "det", "backups_ratio_pi_over_vi", "ratio backups PI/VI - determinista")
    _heat(axes[1, 1], "sto", "backups_ratio_pi_over_vi", "ratio backups PI/VI - estocastico")
    fig.suptitle("[PI vs VI] Ratios PI/VI en la malla gamma x theta  (>1 => PI mas costoso, <1 => VI mas costoso)")
    _save(fig, name)


def make_all_plots(rows):
    plot_value_overlay(rows, "1_value_overlay.png")
    plot_value_diff(rows, "1_value_diff.png")

    plot_policy_agreement(rows, "2_policy_agreement_vs_gamma.png")
    plot_policy_agreement_vs_tol(rows, "2_policy_agreement_vs_tol.png")
    plot_policy_grids_side_by_side(rows, "det", BASE_GAMMA, BASE_TOL, "2_policy_grids_side_by_side_base.png")
    # un caso con tol muy laxa, donde SI es esperable ver discrepancias
    plot_policy_grids_side_by_side(rows, "det", BASE_GAMMA, 10.0, "2_policy_grids_side_by_side_loose_tol.png")

    plot_time_comparison(rows, "3_time_vs_gamma.png")
    plot_time_comparison_vs_tol(rows, "3_time_vs_tol.png")
    plot_backups_comparison(rows, "3_backups_vs_gamma.png")
    plot_heatmaps_ratio(rows, "3_joint_heatmaps_ratio.png")
    print(f"Graficas guardadas en: {PLOTS_DIR}")


if __name__ == "__main__":
    print("Ejecutando comparacion Policy Iteration vs Value Iteration...")
    print(f"GAMMAS={GAMMAS}\nTOLS={TOLS}\nbase: gamma={BASE_GAMMA}, tol={BASE_TOL:g} | "
          f"r_step={R_STEP}, r_obs={R_OBSTACLE}, r_goal={R_GOAL}")
    rows = run_grid()
    save_data(rows)
    print_headline_summary(rows)
    make_all_plots(rows)
    print("\nListo. Revisa 'results_vi_vs_pi/' (datos en 'data/', graficas en 'plots/').")
