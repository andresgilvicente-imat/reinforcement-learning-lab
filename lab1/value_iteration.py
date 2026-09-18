"""Value Iteration — versión ALUMNO.

Rellena únicamente el bloque marcado con ``START CODE HERE`` /
``END CODE HERE``. No modifiques nada fuera de ese bloque: los argumentos que
recibe la función son suficientes para desarrollar el algoritmo.

Convención del modelo de transiciones del entorno:
    P[s][a] -> [(prob, next_state, reward, terminal), ...]
"""

import numpy as np


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Algoritmo Value Iteration.

    Debe aplicar la actualización de optimalidad de Bellman

        V(s) <- max_a sum_{s',r} p(s', r | s, a) [ r + gamma * V(s') ]

    hasta que la variación máxima entre dos barridos sea menor que ``tol``, y
    devolver también la política greedy asociada a la V obtenida.

    Args:
        P (dict): modelo de transiciones del entorno.
        nS (int): número de estados.
        nA (int): número de acciones.
        gamma (float): factor de descuento.
        tol (float): umbral de convergencia theta sobre max|V_nuevo - V|.

    Returns:
        tuple: ``(V, policy)`` con la función de valor y la política óptimas.
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    # ============================================================
    # START CODE HERE
    #
    # TODO: en cada barrido y para cada estado, calcula Q(s, a) para todas las
    #       acciones, asigna a V[s] el máximo y a policy[s] el argmax. Repite
    #       hasta que la mayor variación de un barrido sea menor que tol.
    # HINT: a diferencia de Policy Iteration, aquí no hace falta evaluar una
    #       política completa antes de mejorarla: ambas cosas ocurren en el
    #       mismo barrido.

    while True:
        delta = 0
        for state in range(nS):
            Q = np.zeros(nA)
            v  = V[state]
            for action in range(nA): 
                Q[action] = sum(probability * (reward + gamma * V[next_state]) 
                                for probability, next_state, reward, terminal in P[state][action]) # Al ser determinista realmente solo hay 1 acción, pero puede ser que sea stochastic
                                                                                                   # puede haber varios sitios a los que llegar con esa acción

            V[state] = max(Q)
            policy[state] = np.argmax(Q)

            delta = max(delta, abs(v-V[state]))

        if delta < tol:
            break
    # END CODE HERE
    # ============================================================

    return V, policy
