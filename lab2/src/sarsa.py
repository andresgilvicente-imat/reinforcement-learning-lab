"""Algoritmo SARSA tabular — versión ALUMNO.

Rellena los dos bloques marcados con ``START CODE HERE`` / ``END CODE HERE`` y
borra el ``raise NotImplementedError``. No modifiques nada fuera de esos
bloques: las firmas ya te dan todo lo que necesitas.

SARSA es TD(0) **on-policy**: el target usa la acción que realmente vas a
ejecutar en el estado siguiente, muestreada con la misma política
epsilon-greedy que estás siguiendo.

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
"""

from __future__ import annotations

import numpy as np

from entrenamiento import entrena, es_exito  # noqa: F401  (es_exito, en el hueco)


def epsilon_greedy(Q, state, epsilon, rng):
    """Elige una acción epsilon-greedy sobre ``Q[state]``.

    Con probabilidad ``epsilon`` devuelve una acción aleatoria y con
    probabilidad ``1 - epsilon`` la de mayor ``Q[state, a]``.

    Args:
        Q (np.ndarray): tabla Q de forma ``(nS, nA)``.
        state (int): estado actual.
        epsilon (float): probabilidad de explorar.
        rng (np.random.Generator): generador aleatorio del entrenamiento.
            Usa ``rng.random()`` y ``rng.integers(...)``, no ``np.random``:
            así el entrenamiento es reproducible con la misma semilla.

    Returns:
        int: acción elegida.
    """
    # ============================================================
    # START CODE HERE
    nA = Q.shape[-1]
    if rng.random() < epsilon:
        random_action = rng.integers(0,nA)
        return random_action
    else:
        max_action = np.argmax(Q[state])
        return max_action

    # END CODE HERE
    # ============================================================


def episodio_sarsa(env, Q, alpha, gamma, epsilon, rng):
    """Ejecuta un episodio completo de SARSA actualizando ``Q`` in situ.

    Args:
        env: entorno Gymnasium.
        Q (np.ndarray): tabla Q, que debes modificar durante el episodio.
        alpha (float): tasa de aprendizaje.
        gamma (float): factor de descuento.
        epsilon (float): probabilidad de explorar.
        rng (np.random.Generator): generador aleatorio.

    Returns:
        tuple: ``(retorno, pasos, exito)`` del episodio. ``retorno`` es la suma
        de recompensas sin descontar, ``pasos`` el número de transiciones y
        ``exito`` si el episodio terminó en el objetivo (usa ``es_exito``,
        porque chocar también termina el episodio).
    """
    # ============================================================
    # START CODE HERE
    # Pista: la estructura de un episodio de SARSA es
    #   1. state, _ = env.reset() y elige la primera acción con epsilon_greedy.
    #   2. Repite: env.step(action); si el episodio no ha terminado, elige
    #      a' con epsilon_greedy desde s' y usa Q[s', a'] en el target; si ha
    #      terminado, el target es solo la recompensa (no hay Q(s', a')).
    #   3. Actualiza Q[state, action] con la regla de SARSA y avanza
    #      s <- s', a <- a'.
    #   4. Acumula el retorno y los pasos, y devuelve los tres valores.

    state, _ = env.reset()
    action = epsilon_greedy(Q,state,epsilon,rng)
    t = 0
    r = 0
    success = False
    while True:
        next_state, reward, terminated, _, _ = env.step(action)
        t += 1
        r += reward # without discount 

        if not terminated:
            next_action = epsilon_greedy(Q,next_state,epsilon,rng)

            Q[state,action] += alpha * (reward+gamma*Q[next_state,next_action]- Q[state,action])

        else:

            Q[state,action] += alpha * (reward- Q[state,action])

            success = es_exito(env,reward,terminated)
            break

        action = next_action
        state = next_state

    return (r,t,success)
    # END CODE HERE
    # ============================================================
    


def sarsa(
    env,
    num_episodes,
    alpha,
    gamma,
    epsilon,
    q_init=0.0,
    seed=None,
    track_states=(),
):
    """Entrena un agente con SARSA tabular.

    Args y valor devuelto: ver :func:`entrenamiento.entrena`.
    """
    return entrena(
        env,
        episodio_sarsa,
        num_episodes,
        alpha,
        gamma,
        epsilon,
        q_init=q_init,
        seed=seed,
        track_states=track_states,
    )


def greedy_policy(Q):
    """Política determinista derivada de Q (argmax por estado)."""
    return np.argmax(Q, axis=1).astype(int)
