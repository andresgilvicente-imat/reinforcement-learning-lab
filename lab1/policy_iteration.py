"""Policy Iteration — versión ALUMNO.

Rellena únicamente los bloques marcados con ``START CODE HERE`` /
``END CODE HERE``. No modifiques nada fuera de esos bloques: los argumentos
que recibe cada función son suficientes para desarrollar el algoritmo.

Orden recomendado de trabajo:
    1. ``policy_evaluation``
    2. ``policy_improvement``
    3. ``policy_iteration``

Convención del modelo de transiciones del entorno:
    P[s][a] -> [(prob, next_state, reward, terminal), ...]
donde
    prob (float)       probabilidad de la transición
    next_state (int)   estado al que se transita, en [0, nS - 1]
    reward (float)     recompensa de la transición
    terminal (bool)    True si next_state es terminal (obstáculo u objetivo)
"""

import numpy as np


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evalúa iterativamente la función de valor de una política determinista.

    Debe aplicar la ecuación de Bellman de evaluación

        V(s) <- sum_{s',r} p(s', r | s, pi(s)) [ r + gamma * V(s') ]

    hasta que la variación máxima entre dos barridos sea menor que ``tol``.

    Args:
        P (dict): modelo de transiciones del entorno.
        nS (int): número de estados.
        nA (int): número de acciones.
        policy (np.ndarray[nS]): política a evaluar (una acción por estado)
        gamma (float): factor de descuento.
        tol (float): umbral de convergencia theta sobre max|V_nuevo - V|.

    Returns:
        np.ndarray[nS]: función de valor V de la política.
    """
    V = np.zeros(nS)

    # ============================================================
    # START CODE HERE
    #
    # TODO: bucle que repita barridos sobre todos los estados actualizando
    #       V[s] con la ecuación de Bellman para la acción policy[s], y que
    #       termine cuando la mayor variación de un barrido sea menor que tol.
    # HINT: la acción que dicta la política en el estado s es policy[s], y sus
    #       transiciones posibles son P[s][policy[s]].
    while True:
        delta = 0.0

        for state in range(nS):
            old_value = V[state]
            action = int(policy[state]) # viene determinado ya por la policy es obligatoria, politica arbitraria
            V[state] = sum(
                probability * (reward + gamma * V[next_state])
                for probability, next_state, reward, terminal in P[state][action]
            )
            delta = max(delta, abs(old_value - V[state]))

        if delta < tol:
            break

    # END CODE HERE
    # ============================================================

    return V


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Devuelve la política greedy respecto a una función de valor.

    Debe calcular, para cada estado,

        Q(s, a) = sum_{s',r} p(s', r | s, a) [ r + gamma * V(s') ]

    y elegir la acción que la maximiza.

    Args:
        P (dict): modelo de transiciones del entorno.
        nS (int): número de estados.
        nA (int): número de acciones.
        value_from_policy (np.ndarray[nS]): función de valor de partida.
        gamma (float): factor de descuento.

    Returns:
        np.ndarray[nS]: política determinista mejorada (un entero por estado).
    """
    new_policy = np.zeros(nS, dtype=int)

    # ============================================================
    # START CODE HERE
    #
    # TODO: para cada estado, calcula Q(s, a) para todas las acciones a partir
    #       de P y de value_from_policy, y guarda en new_policy[s] la acción
    #       que maximiza Q.
    # HINT: np.argmax devuelve el índice del máximo.
    for state in range(nS):
        action_values = np.zeros(nA)

        for action in range(nA):
            action_values[action] = sum(
                probability * (reward + gamma * value_from_policy[next_state])
                for probability, next_state, reward, terminal in P[state][action]
            )

        new_policy[state] = int(np.argmax(action_values))

    return new_policy
    # END CODE HERE
    # ============================================================


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Algoritmo Policy Iteration.

    Debe alternar :func:`policy_evaluation` y :func:`policy_improvement` hasta
    que la política deje de cambiar entre dos iteraciones consecutivas.

    Args:
        P (dict): modelo de transiciones del entorno.
        nS (int): número de estados.
        nA (int): número de acciones.
        gamma (float): factor de descuento.
        tol (float): umbral de convergencia usado en la evaluación.

    Returns:
        tuple: ``(V, policy)`` con la función de valor y la política óptimas.
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    # ============================================================
    # START CODE HERE
    #
    # TODO: repite [evaluar la política -> mejorarla] hasta que la política
    #       nueva coincida con la anterior, y devuelve la V de esa política.
    # HINT: np.array_equal compara dos políticas.

    while True:
        V = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, V, gamma)

        if np.array_equal(policy, new_policy):
            policy = new_policy
            break

        policy = new_policy
    # END CODE HERE
    # ============================================================

    return V, policy
