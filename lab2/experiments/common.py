"""Utilidades compartidas por los experimentos de la Práctica 2.

* Importa el código real de ``lab2/src`` (sin modificarlo).
* Calcula referencias teóricas a partir del modelo ``P`` del entorno. El modelo
  solo se usa aquí, para comparar; nunca se pasa a SARSA ni a Q-learning.
* Ejecuta entrenamientos instrumentados que reutilizan ``episodio_sarsa`` y
  ``episodio_q_learning`` tal cual, con la misma semilla y el mismo orden de
  llamadas que ``entrenamiento.entrena`` (``validate_algorithms.py`` comprueba
  que los resultados coinciden bit a bit).
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

LAB2 = Path(__file__).resolve().parents[1]
SRC = LAB2 / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gymnasium as gym  # noqa: E402
from gymnasium.envs.registration import register, registry  # noqa: E402

from env import (  # noqa: E402
    INITIAL_STATES,
    N_NON_TERMINAL,
    R_GOAL,
    R_OBSTACLE,
    R_STEP,
    TERMINAL_STATE,
)
from q_learning import episodio_q_learning, q_learning  # noqa: E402
from sarsa import episodio_sarsa, greedy_policy, sarsa  # noqa: E402

ENV_ID = "JumpToTheGoalEnv-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="env:JumpToTheGoalEnv")

GAMMA = 0.9
NS, NA = N_NON_TERMINAL + 1, 2
UR, DR = 0, 1
ACTION_SYMBOL = {UR: "↗", DR: "↘"}
ACTION_TEX = {UR: r"\nearrow", DR: r"\searrow"}

RESULTS = LAB2 / "results"
FIGURES = LAB2 / "figures"

EPISODE_FN = {"sarsa": episodio_sarsa, "q_learning": episodio_q_learning}
TRAIN_FN = {"sarsa": sarsa, "q_learning": q_learning}
ALGO_LABEL = {"sarsa": "SARSA", "q_learning": "Q-learning"}


def make_env(r_step=R_STEP, r_obstacle=R_OBSTACLE, r_goal=R_GOAL, initial_state=None):
    """Mismo entorno que crea ``main.py`` (determinista, s_init aleatorio)."""
    return gym.make(
        ENV_ID,
        deterministic=True,
        r_step=r_step,
        r_obstacle=r_obstacle,
        r_goal=r_goal,
        initial_state=initial_state,
    )


# --------------------------------------------------------------------------
# Referencias teóricas (usan el modelo P; solo para análisis)
# --------------------------------------------------------------------------
def _backup(P, Q, gamma, eps):
    """Operador de Bellman epsilon-soft óptimo.

    T_eps Q(s,a) = sum p [r + gamma * ((1-eps) max_a' Q(s',a') + eps * mean_a' Q(s',a'))]

    Con eps = 0 es el operador de optimalidad de Bellman (punto fijo q*). Con
    eps > 0 su punto fijo es el valor de la mejor política epsilon-greedy, que
    es lo que estima SARSA con epsilon constante. Es una contracción de módulo
    gamma, así que la iteración converge desde cualquier Q inicial.
    """
    new = np.zeros_like(Q)
    for s in range(NS):
        for a in range(NA):
            total = 0.0
            for p, s2, r, term in P[s][a]:
                if term or s2 == TERMINAL_STATE:
                    boot = 0.0
                else:
                    boot = (1 - eps) * Q[s2].max() + eps * Q[s2].mean()
                total += p * (r + gamma * boot)
            new[s, a] = total
    new[TERMINAL_STATE] = 0.0
    return new


def soft_optimal_q(P, gamma=GAMMA, eps=0.0, tol=1e-13, max_iter=10_000):
    """Punto fijo de T_eps (q* si eps = 0). Devuelve (Q, iteraciones)."""
    Q = np.zeros((NS, NA))
    for it in range(1, max_iter + 1):
        new = _backup(P, Q, gamma, eps)
        if np.max(np.abs(new - Q)) < tol:
            return new, it
        Q = new
    raise RuntimeError("no converge")


def q_star(P, gamma=GAMMA):
    return soft_optimal_q(P, gamma, 0.0)[0]


def optimal_action_sets(Q, atol=1e-9):
    """Conjunto de acciones óptimas por estado (empates incluidos)."""
    return [set(np.flatnonzero(Q[s] >= Q[s].max() - atol)) for s in range(NS)]


def successors(P, s, a):
    return [(s2, r, term) for p, s2, r, term in P[s][a] if p > 0]


def reachable_states(P, starts=INITIAL_STATES):
    """Estados no terminales alcanzables desde ``starts`` con alguna política."""
    seen, queue = set(starts), deque(starts)
    while queue:
        s = queue.popleft()
        for a in range(NA):
            for s2, _, term in successors(P, s, a):
                if not term and s2 not in seen:
                    seen.add(s2)
                    queue.append(s2)
    return sorted(seen)


def goal_reachable(P):
    """Estados desde los que alguna secuencia de acciones llega al objetivo."""
    good = set()
    changed = True
    while changed:
        changed = False
        for s in range(N_NON_TERMINAL):
            if s in good:
                continue
            for a in range(NA):
                for s2, r, term in successors(P, s, a):
                    if (term and r > 0) or (not term and s2 in good):
                        good.add(s)
                        changed = True
    return sorted(good)


def stationary_sd_sarsa(P, Qeps, s, a, alpha, eps, gamma=GAMMA):
    """Desviación típica estacionaria aproximada de Q(s,a) en SARSA.

    Se supone que las Q del sucesor ya están en su punto fijo (Qeps) y que la
    única aleatoriedad del target es la acción a' ~ eps-greedy. Entonces
    Q_{n+1} = (1-alpha) Q_n + alpha X_n es un AR(1) con varianza estacionaria
    alpha * Var(X) / (2 - alpha).
    """
    (s2, r, term), = successors(P, s, a)
    if term:
        return 0.0, 0.0
    q2 = Qeps[s2]
    greedy = int(np.argmax(q2))
    probs = np.full(NA, eps / NA)
    probs[greedy] += 1 - eps
    mean_t = r + gamma * probs @ q2
    var_t = gamma**2 * (probs @ (q2 - probs @ q2) ** 2)
    return float(np.sqrt(alpha * var_t / (2 - alpha))), float(np.sqrt(var_t))


def eps_greedy_values(P, policy, eps, gamma=1.0, n_iter=200):
    """V de la política eps-greedy alrededor de ``policy`` (evaluación exacta).

    Con gamma = 1 da el retorno esperado SIN descuento, comparable con el
    retorno de entrenamiento que registra ``entrenamiento.py``.
    """
    V = np.zeros(NS)
    for _ in range(n_iter):
        for s in range(NS - 1):
            probs = np.full(NA, eps / NA)
            probs[policy[s]] += 1 - eps
            V[s] = sum(
                probs[a] * sum(p * (r + gamma * (0.0 if t else V[s2])) for p, s2, r, t in P[s][a])
                for a in range(NA)
            )
    return V


def expected_training_return(P, policy, eps):
    """Retorno medio sin descuento desde s_init (uniforme) con eps-greedy."""
    V = eps_greedy_values(P, policy, eps)
    return float(np.mean([V[s] for s in INITIAL_STATES]))


# --------------------------------------------------------------------------
# Entrenamiento instrumentado
# --------------------------------------------------------------------------
class Recorder(gym.Wrapper):
    """Registra cada transición sin alterar el entorno ni consumir azar."""

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma
        self.ep_pairs = []
        self.disc_return = 0.0
        self.start = None
        self._t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ep_pairs = []
        self.disc_return = 0.0
        self.start = int(obs)
        self._t = 0
        return obs, info

    def step(self, action):
        s = int(self.env.unwrapped.state)
        self.ep_pairs.append((s, int(action)))
        obs, r, term, trunc, info = self.env.step(action)
        self.disc_return += (self.gamma**self._t) * r
        self._t += 1
        return obs, r, term, trunc, info


def greedy_rollout(env, Q, start, max_steps=20):
    """Episodio con la política greedy (epsilon = 0) desde ``start``."""
    s, _ = env.reset(options={"initial_state": start})
    ret, success = 0.0, False
    for _ in range(max_steps):
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
        ret += r
        if term or trunc:
            success = r == env.unwrapped.r_goal
            break
    return ret, success


def run_training(
    algo,
    episodes,
    alpha,
    epsilon,
    q_init=0.0,
    seed=0,
    gamma=GAMMA,
    rewards=(R_STEP, R_OBSTACLE, R_GOAL),
    eval_every=20,
    eps_schedule=None,
    alpha_schedule=None,
):
    """Entrena con la función de episodio real y registra todo lo necesario.

    Reproduce ``entrenamiento.entrena`` (misma inicialización de Q, del rng y
    del entorno). ``eps_schedule(k)`` / ``alpha_schedule(k)`` permiten las
    extensiones con exploración o tasa de aprendizaje decrecientes.
    """
    env = Recorder(make_env(*rewards), gamma)
    eval_env = make_env(*rewards)
    episodio = EPISODE_FN[algo]

    Q = np.full((NS, NA), float(q_init), dtype=float)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    q_hist = np.empty((episodes, NS, NA))
    visits = np.zeros((NS, NA), dtype=np.int64)
    visits_hist = np.empty((episodes, NS, NA), dtype=np.int32)
    returns = np.empty(episodes)
    disc_returns = np.empty(episodes)
    steps = np.empty(episodes, dtype=np.int32)
    success = np.empty(episodes, dtype=bool)
    starts = np.empty(episodes, dtype=np.int8)
    n_eval = episodes // eval_every + 1
    eval_ep = np.empty(n_eval, dtype=np.int32)
    eval_ret = np.empty((n_eval, len(INITIAL_STATES)))
    eval_succ = np.empty((n_eval, len(INITIAL_STATES)), dtype=bool)

    def evaluate(j, k):
        eval_ep[j] = k
        for i, s0 in enumerate(INITIAL_STATES):
            eval_ret[j, i], eval_succ[j, i] = greedy_rollout(eval_env, Q, s0)

    evaluate(0, 0)
    for k in range(episodes):
        eps_k = eps_schedule(k) if eps_schedule else epsilon
        alpha_k = alpha_schedule(k) if alpha_schedule else alpha
        ret, n, ok = episodio(env, Q, alpha_k, gamma, eps_k, rng)
        returns[k], steps[k], success[k] = ret, n, ok
        disc_returns[k] = env.disc_return
        starts[k] = env.start
        for s, a in env.ep_pairs:
            visits[s, a] += 1
        visits_hist[k] = visits
        q_hist[k] = Q
        if (k + 1) % eval_every == 0:
            evaluate((k + 1) // eval_every, k + 1)

    env.close()
    eval_env.close()
    return {
        "Q": Q.copy(),
        "q_hist": q_hist,
        "visits": visits,
        "visits_hist": visits_hist,
        "returns": returns,
        "disc_returns": disc_returns,
        "steps": steps,
        "cum_steps": np.cumsum(steps),
        "success": success,
        "starts": starts,
        "eval_ep": eval_ep,
        "eval_ret": eval_ret,
        "eval_succ": eval_succ,
    }


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def bootstrap_ci(x, n_boot=10_000, level=0.95, seed=0):
    """IC bootstrap percentil de la media."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def sign_flip_pvalue(d, n_perm=100_000, seed=0):
    """p-valor bilateral del test de permutación pareado (cambio de signo)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n_perm + 1))
