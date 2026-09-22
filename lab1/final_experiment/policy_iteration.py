"""Policy Iteration — versión ALUMNO (completada).

Convención del modelo de transiciones del entorno:
    P[s][a] -> [(prob, next_state, reward, terminal), ...]
"""

import numpy as np


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evalúa iterativamente la función de valor de una política determinista."""
    V = np.zeros(nS)

    while True:
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

    return V


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Devuelve la política greedy respecto a una función de valor."""
    new_policy = np.zeros(nS, dtype=int)

    for state in range(nS):
        action_values = np.zeros(nA)

        for action in range(nA):
            action_values[action] = sum(
                probability * (reward + gamma * value_from_policy[next_state])
                for probability, next_state, reward, terminal in P[state][action]
            )

        new_policy[state] = int(np.argmax(action_values))

    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Algoritmo Policy Iteration."""
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    while True:
        V = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, V, gamma)

        converged = np.array_equal(policy, new_policy)
        policy = new_policy

        if converged:
            break

    return V, policy
