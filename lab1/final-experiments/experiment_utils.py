"""Utilidades para las dos figuras deterministas repetidas en 3.3 y 4.3."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lab1.src.env import INITIAL_STATES, JumpToTheGoalEnv  # noqa: E402
from lab1.src.policy_iteration import policy_improvement, policy_iteration  # noqa: E402
from lab1.src.value_iteration import value_iteration  # noqa: E402

BASE_GAMMA = 0.9
BASE_TOL = 1e-3
GAMMAS = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
TOLERANCES = (10.0, 3.0, 1.0, 0.3, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
TIMING_REPEATS = 7


def policy_evaluation_instrumented(P, nS, policy, gamma, tol):
    """Evalúa una política y devuelve también el número de barridos."""
    values = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for state in range(nS):
            old_value = values[state]
            action = int(policy[state])
            values[state] = sum(
                probability * (reward + gamma * values[next_state])
                for probability, next_state, reward, _ in P[state][action]
            )
            delta = max(delta, abs(old_value - values[state]))
        if delta < tol:
            return values, sweeps


def policy_iteration_instrumented(P, nS, nA, gamma, tol):
    """Ejecuta PI y registra iteraciones externas y barridos de evaluación."""
    policy = np.zeros(nS, dtype=int)
    outer_iterations = 0
    evaluation_sweeps = 0
    while True:
        outer_iterations += 1
        values, sweeps = policy_evaluation_instrumented(
            P, nS, policy, gamma, tol
        )
        evaluation_sweeps += sweeps
        new_policy = policy_improvement(P, nS, nA, values, gamma)
        converged = np.array_equal(policy, new_policy)
        policy = new_policy
        if converged:
            return values, policy, outer_iterations, evaluation_sweeps


def value_iteration_instrumented(P, nS, nA, gamma, tol):
    """Ejecuta VI y registra el número de barridos."""
    values = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for state in range(nS):
            old_value = values[state]
            values[state] = max(
                sum(
                    probability * (reward + gamma * values[next_state])
                    for probability, next_state, reward, _ in P[state][action]
                )
                for action in range(nA)
            )
            delta = max(delta, abs(old_value - values[state]))
        if delta < tol:
            break

    policy = np.array(
        [
            max(
                range(nA),
                key=lambda action: sum(
                    probability * (reward + gamma * values[next_state])
                    for probability, next_state, reward, _ in P[state][action]
                ),
            )
            for state in range(nS)
        ]
    )
    return values, policy, sweeps


def best_time_ms(function, *args, repeats=TIMING_REPEATS):
    """Devuelve el menor tiempo de varias ejecuciones, en milisegundos."""
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def _configurations():
    configurations = [(gamma, BASE_TOL) for gamma in GAMMAS]
    configurations.extend(
        (BASE_GAMMA, tol) for tol in TOLERANCES if tol != BASE_TOL
    )
    return configurations


def get_result(rows, gamma, tol):
    """Selecciona una configuración mediante comparación numérica robusta."""
    for row in rows:
        same_gamma = np.isclose(row["gamma"], gamma, rtol=0.0, atol=1e-15)
        same_tol = np.isclose(row["tol"], tol, rtol=0.0, atol=1e-15)
        if same_gamma and same_tol:
            return row
    raise KeyError((gamma, tol))


def run_comparison():
    """Calcula las dos secciones deterministas usadas por ambas figuras."""
    env = JumpToTheGoalEnv(deterministic=True, prob=1.0)
    rows = []
    for gamma, tol in _configurations():
        values_pi, policy_pi, pi_outer, pi_sweeps = policy_iteration_instrumented(
            env.P, env.nS, env.nA, gamma, tol
        )
        values_vi, policy_vi, vi_sweeps = value_iteration_instrumented(
            env.P, env.nS, env.nA, gamma, tol
        )
        official_pi = policy_iteration(env.P, env.nS, env.nA, gamma, tol)
        official_vi = value_iteration(env.P, env.nS, env.nA, gamma, tol)
        if not (
            np.allclose(values_pi, official_pi[0])
            and np.array_equal(policy_pi, official_pi[1])
            and np.allclose(values_vi, official_vi[0])
            and np.array_equal(policy_vi, official_vi[1])
        ):
            raise AssertionError("La instrumentación no coincide con los algoritmos base.")

        rows.append(
            {
                "gamma": gamma,
                "tol": tol,
                "V_pi": values_pi,
                "V_vi": values_vi,
                "policy_pi": policy_pi,
                "policy_vi": policy_vi,
                "max_V_diff": float(np.max(np.abs(values_pi - values_vi))),
                "policies_equal": bool(np.array_equal(policy_pi, policy_vi)),
                "pi_outer": pi_outer,
                "pi_eval_sweeps": pi_sweeps,
                "vi_sweeps": vi_sweeps,
                "pi_evaluations": int(
                    pi_sweeps * env.nS + pi_outer * env.nS * env.nA
                ),
                "vi_evaluations": int(vi_sweeps * env.nS * env.nA),
                "pi_time_ms": best_time_ms(
                    policy_iteration, env.P, env.nS, env.nA, gamma, tol
                ),
                "vi_time_ms": best_time_ms(
                    value_iteration, env.P, env.nS, env.nA, gamma, tol
                ),
            }
        )
    env.close()
    return rows


def _save_figure(fig, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_value_overlay(rows, output_dir):
    """Superpone los valores de PI y VI al variar gamma."""
    selected = [get_result(rows, gamma, BASE_TOL) for gamma in GAMMAS]
    fig, ax = plt.subplots(figsize=(8, 5))
    for state in INITIAL_STATES:
        ax.plot(
            GAMMAS,
            [row["V_pi"][state] for row in selected],
            marker="o",
            linewidth=2.5,
            alpha=0.65,
            label=f"PI s{state}",
        )
        ax.plot(
            GAMMAS,
            [row["V_vi"][state] for row in selected],
            marker="x",
            linewidth=1.2,
            linestyle="--",
            label=f"VI s{state}",
        )
    ax.axvline(BASE_GAMMA, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(r"Factor de descuento $\gamma$")
    ax.set_ylabel(r"$V(s)$ del estado inicial")
    ax.set_title(
        rf"PI frente a VI: valores en el entorno determinista "
        rf"($\theta={BASE_TOL:g}$)"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return _save_figure(fig, output_dir, "1_value_overlay_deterministic.png")


def _draw_times(ax, x_values, selected, xlabel):
    ax.plot(
        x_values,
        [row["pi_time_ms"] for row in selected],
        marker="o",
        color="tab:purple",
        label="Policy Iteration",
    )
    ax.plot(
        x_values,
        [row["vi_time_ms"] for row in selected],
        marker="s",
        color="tab:green",
        label="Value Iteration",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tiempo de convergencia (ms)")
    ax.grid(alpha=0.3)


def plot_time_comparison(rows, output_dir):
    """Compara tiempos frente a gamma y theta en el entorno determinista."""
    gamma_rows = [get_result(rows, gamma, BASE_TOL) for gamma in GAMMAS]
    tolerance_rows = [get_result(rows, BASE_GAMMA, tol) for tol in TOLERANCES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    _draw_times(axes[0], GAMMAS, gamma_rows, r"Factor de descuento $\gamma$")
    axes[0].axvline(BASE_GAMMA, color="gray", linestyle=":", linewidth=1)
    axes[0].set_title(rf"Variacion de $\gamma$ ($\theta={BASE_TOL:g}$)")
    axes[0].legend()
    _draw_times(axes[1], TOLERANCES, tolerance_rows, r"Tolerancia $\theta$")
    axes[1].set_xscale("log")
    axes[1].axvline(BASE_TOL, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title(rf"Variacion de $\theta$ ($\gamma={BASE_GAMMA:g}$)")
    fig.suptitle("PI frente a VI: tiempos medidos (entorno determinista)")
    return _save_figure(
        fig, output_dir, "3_time_gamma_tol_deterministic.png"
    )


def validate_results(rows, output_dir, expected_figures):
    """Comprueba cifras citadas y el conjunto exacto de figuras."""
    assert len(rows) == len(GAMMAS) + len(TOLERANCES) - 1
    assert all(row["max_V_diff"] == 0.0 for row in rows)
    assert all(row["policies_equal"] for row in rows)

    base = get_result(rows, BASE_GAMMA, BASE_TOL)
    assert base["pi_outer"] == 3
    assert base["pi_eval_sweeps"] == 11
    assert base["vi_sweeps"] == 4
    assert base["pi_evaluations"] == 357
    assert base["vi_evaluations"] == 168
    assert base["pi_time_ms"] < 1.0
    assert base["vi_time_ms"] < 1.0

    ratio = base["pi_time_ms"] / base["vi_time_ms"]
    assert 1.2 < ratio < 4.0
    actual_figures = {path.name for path in Path(output_dir).glob("*.png")}
    assert actual_figures == set(expected_figures)
    return {
        "pi_time_ms": base["pi_time_ms"],
        "vi_time_ms": base["vi_time_ms"],
        "time_ratio_pi_over_vi": ratio,
        "figures": len(actual_figures),
    }
