"""Value Iteration — versión ALUMNO (completada).

Convención del modelo de transiciones del entorno:
    P[s][a] -> [(prob, next_state, reward, terminal), ...]
"""

import numpy as np


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Algoritmo Value Iteration."""
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    while True:

        delta = 0

        for state in range(nS):

            old_value = V[state]
            V[state] = max(
                sum(probability * (reward + gamma * V[next_state])
                    for probability, next_state, reward, terminal in P[state][action]
                    )
                for action in range(nA)
            )

            delta = max(delta, abs(old_value - V[state]))

        if delta < tol:
            break

    policy = np.array([
        max(range(nA), key=lambda action:
            sum(probability * (reward + gamma * V[next_state])
                for probability, next_state, reward, terminal in P[state][action]))
        for state in range(nS)
    ])

    return V, policy
