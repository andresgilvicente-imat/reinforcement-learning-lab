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
    delta = np.inf

    while delta > tol:
        for i in range(nS):
            v = V[i]

            accion = policy[i] # 0 o 1, en funcion del env, arriba derecha o abajo derecha
            # P[i][accion] -- nos da una lista, al ser determinista solo hay 1 elemento
            prob,next_state, reward, terminal = P[i][accion][0] # P[i][accion] = (prob,next_state, reward, terminal)
            V[i] = prob * (reward + gamma*V[next_state])

            delta = min(delta, np.abs(v - V[i]))

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
    policy_stable = True
    for i in range(nS):
        probs = []
        actions = []
        for j in P[i]:
            prob, next_state, reward, terminal = P[i][j][0]
            probs.append(prob)
            actions.append(j)

        old_action = actions[np.argmax(probs)]
        rewards = []
        actions = []
        for k in range(nS):
            if k != i:
                for h in P[i]:
                    prob, next_state, reward, terminal = P[i][h][0]

                    final_reward = reward + gamma * value_from_policy[next_state]

                    rewards.append(final_reward)
                    actions.append(h)

        new_policy[i] = actions[np.argmax(rewards)]

        if old_action != new_policy[i]:
            policy_stable = False

    if policy_stable:
        return value_from_policy, new_policy
    else:
        return False, new_policy
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
        past_policy = policy
        V = policy_evaluation(P,nS,nA,policy,gamma,tol)
        value_from_policy, policy = policy_improvement(P,nS,nA,value_from_policy=V,gamma=gamma)
        
        if value_from_policy:
            print("====================================")
            print("V", V)
            print("====================================")

            if np.array_equal(policy,past_policy):
                break
    # END CODE HERE
    # ============================================================

    return V, policy
