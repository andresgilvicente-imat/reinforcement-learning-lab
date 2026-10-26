"""Validación de las implementaciones de SARSA y Q-learning (Práctica 2).

Uso (desde lab2/experiments):  python validate_algorithms.py

Comprueba con el código real de ``lab2/src``:
  1. epsilon_greedy: frecuencias, desempate y casos límite.
  2. Reglas de actualización sobre un entorno de juguete con azar guionizado:
     SARSA usa Q(s', a') con la a' ejecutada; Q-learning usa max_a' Q(s', a');
     en transición terminal no hay bootstrapping.
  3. El runner instrumentado reproduce bit a bit sarsa() y q_learning().
  4. Con epsilon = 0 SARSA y Q-learning generan exactamente la misma Q.
  5. Entrenar N episodios es el prefijo de entrenar M > N con la misma semilla.
  6. La fila del terminal y los estados inalcanzables no se actualizan.
  7. q* coincide con Policy/Value Iteration de la Práctica 1 y con los valores
     calculados a mano.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np

from common import (
    GAMMA,
    LAB2,
    NA,
    NS,
    RESULTS,
    TERMINAL_STATE,
    TRAIN_FN,
    make_env,
    q_star,
    reachable_states,
    run_training,
)
from q_learning import episodio_q_learning
from sarsa import epsilon_greedy, episodio_sarsa

RESULTS_LINES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    line = f"[{status}] {name}" + (f"  ({detail})" if detail else "")
    print(line)
    RESULTS_LINES.append(line)
    return cond


# ----------------------------------------------------------------- 1
def test_epsilon_greedy():
    rng = np.random.default_rng(123)
    Q = np.zeros((3, 2))
    Q[0] = [1.0, 2.0]  # greedy = 1
    Q[1] = [0.0, 0.0]  # empate
    n = 200_000
    for eps in (0.0, 0.2, 0.5, 1.0):
        picks = np.array([epsilon_greedy(Q, 0, eps, rng) for _ in range(n)])
        p_greedy = np.mean(picks == 1)
        expected = 1 - eps + eps / 2
        tol = 4 * np.sqrt(expected * (1 - expected) / n) + 1e-12
        check(
            f"epsilon_greedy eps={eps}: P(greedy)={p_greedy:.4f}",
            abs(p_greedy - expected) <= tol,
            f"esperado {expected:.4f}",
        )
    picks = [epsilon_greedy(Q, 1, 0.0, rng) for _ in range(1000)]
    check("empate con eps=0 -> siempre la acción 0 (np.argmax)", set(picks) == {0})


# ----------------------------------------------------------------- 2
class ScriptedRng:
    """Sustituye al Generator: devuelve valores guionizados."""

    def __init__(self, randoms, integers):
        self.randoms, self.ints = list(randoms), list(integers)

    def random(self):
        return self.randoms.pop(0)

    def integers(self, low, high=None):
        return self.ints.pop(0)


class ToyEnv:
    """Cadena s0 -> s1 -> T (objetivo, +5). Desde s0 cualquier acción va a s1
    con r=-1; desde s1 cualquier acción termina con +5."""

    class _U:
        r_goal = 5.0

    unwrapped = _U()

    def reset(self, seed=None, options=None):
        self.s = 0
        return 0, {}

    def step(self, a):
        if self.s == 0:
            self.s = 1
            return 1, -1.0, False, False, {}
        self.s = 2
        return 2, 5.0, True, False, {}


def test_update_rules():
    alpha, gamma, eps = 0.5, 0.9, 0.5
    # En s0 se explota (random=0.9 -> acción greedy 0); en s1 se explora
    # (random=0.1) y integers devuelve la acción 1, la NO greedy.
    for name, fn in (("SARSA", episodio_sarsa), ("Q-learning", episodio_q_learning)):
        Q = np.zeros((3, 2))
        Q[1] = [10.0, -10.0]
        rng = ScriptedRng([0.9, 0.1], [1])
        ret, n, ok = fn(ToyEnv(), Q, alpha, gamma, eps, rng)
        q1_used = -10.0 if name == "SARSA" else 10.0
        exp_q00 = 0 + alpha * (-1 + gamma * q1_used - 0)
        exp_q11 = -10.0 + alpha * (5.0 - (-10.0))
        check(
            f"{name}: Q(s0,a0) usa {'Q(s1,a′ ejecutada)' if name == 'SARSA' else 'max Q(s1,·)'}",
            np.isclose(Q[0, 0], exp_q00),
            f"{Q[0, 0]:.4f} vs {exp_q00:.4f}",
        )
        check(
            f"{name}: transición terminal sin bootstrap Q(s1,a1) <- Q + α(r − Q)",
            np.isclose(Q[1, 1], exp_q11),
            f"{Q[1, 1]:.4f} vs {exp_q11:.4f}",
        )
        check(f"{name}: devuelve (retorno, pasos, éxito)", (ret, n, ok) == (4.0, 2, True))
        check(f"{name}: no toca Q(s0,a1) ni Q(s1,a0) ni el terminal",
              Q[0, 1] == 0 and Q[1, 0] == 10.0 and np.all(Q[2] == 0))


# ----------------------------------------------------------------- 3
def test_runner_reproduces_original():
    for algo in ("sarsa", "q_learning"):
        for seed, alpha, eps, q0 in ((0, 0.1, 0.2, 0.0), (7, 0.5, 0.5, 5.0), (3, 0.05, 0.05, -5.0)):
            env = make_env()
            ref = TRAIN_FN[algo](env, 1500, alpha, GAMMA, eps, q_init=q0, seed=seed,
                                 track_states=(6, 13))
            env.close()
            mine = run_training(algo, 1500, alpha, eps, q0, seed)
            same = (
                np.array_equal(ref["Q"], mine["Q"])
                and np.array_equal(ref["retornos"], mine["returns"])
                and np.array_equal(ref["pasos"], mine["steps"])
                and np.array_equal(ref["exitos"], mine["success"])
                and np.array_equal(ref["q_history"], mine["q_hist"][:, [6, 13]])
            )
            check(f"runner == {algo}() original (seed={seed}, α={alpha}, ε={eps}, Q0={q0})", same)


# ----------------------------------------------------------------- 4-6
def test_properties():
    for q0 in (0.0, 5.0):
        s = run_training("sarsa", 3000, 0.1, 0.0, q0, 1)
        q = run_training("q_learning", 3000, 0.1, 0.0, q0, 1)
        check(f"ε=0 ⇒ SARSA ≡ Q-learning (Q0={q0})",
              np.array_equal(s["q_hist"], q["q_hist"]) and np.array_equal(s["returns"], q["returns"]))

    long = run_training("sarsa", 2000, 0.1, 0.2, 0.0, 4)
    short = run_training("sarsa", 500, 0.1, 0.2, 0.0, 4)
    check("500 episodios = prefijo de 2000 (misma semilla)",
          np.array_equal(long["q_hist"][:500], short["q_hist"]))

    P = make_env().unwrapped.P
    unreachable = [s for s in range(NS - 1) if s not in reachable_states(P)]
    for algo in ("sarsa", "q_learning"):
        r = run_training(algo, 5000, 0.1, 0.2, 1.234, 2)
        check(f"{algo}: fila del terminal intacta (Q0)", np.all(r["Q"][TERMINAL_STATE] == 1.234))
        check(f"{algo}: inalcanzables {unreachable} intactos (Q0) y sin visitas",
              np.all(r["Q"][unreachable] == 1.234) and r["visits"][unreachable].sum() == 0)


# ----------------------------------------------------------------- 7
def _load_lab1(name):
    path = LAB2.parent / "lab1" / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lab1_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_q_star():
    P = make_env().unwrapped.P
    qs = q_star(P)
    v_vi, _ = _load_lab1("value_iteration").value_iteration(P, NS, NA, GAMMA, 1e-10)
    v_pi, _ = _load_lab1("policy_iteration").policy_iteration(P, NS, NA, GAMMA, 1e-10)
    check("max_a q* = V de Value Iteration (P1)", np.allclose(qs.max(1), v_vi, atol=1e-8))
    check("max_a q* = V de Policy Iteration (P1)", np.allclose(qs.max(1), v_pi, atol=1e-8))
    for (s, a, v) in ((6, 0, 2.15), (6, 1, 5.0), (13, 0, 2.15), (13, 1, -5.5)):
        check(f"q*(s{s},{'↗' if a == 0 else '↘'}) = {v}", np.isclose(qs[s, a], v))


if __name__ == "__main__":
    test_epsilon_greedy()
    test_update_rules()
    test_runner_reproduces_original()
    test_properties()
    test_q_star()
    n_fail = sum(line.startswith("[FAIL]") for line in RESULTS_LINES)
    out = RESULTS / "summaries" / "validation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(RESULTS_LINES) + f"\n\n{len(RESULTS_LINES) - n_fail} PASS, {n_fail} FAIL\n",
                   encoding="utf-8")
    print(f"\n{len(RESULTS_LINES) - n_fail} PASS, {n_fail} FAIL -> {out}")
    sys.exit(1 if n_fail else 0)
