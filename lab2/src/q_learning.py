"""Algoritmo Q-learning tabular — versión ALUMNO.

Rellena el bloque marcado con ``START CODE HERE`` / ``END CODE HERE`` y borra
el ``raise NotImplementedError``. No modifiques nada fuera del bloque.

Q-learning es TD(0) **off-policy**: te comportas con epsilon-greedy, pero el
target usa el máximo sobre las acciones del estado siguiente, no la acción que
vas a ejecutar de verdad.

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (te hará falta en el hueco)

from entrenamiento import entrena, es_exito  # noqa: F401  (es_exito, en el hueco)
from sarsa import epsilon_greedy  # noqa: F401  (reutiliza la de sarsa.py)


def episodio_q_learning(env, Q, alpha, gamma, epsilon, rng):
    """Ejecuta un episodio completo de Q-learning actualizando ``Q`` in situ.

    Args y valor devuelto: ver :func:`sarsa.episodio_sarsa`.
    """
    # ============================================================
    # START CODE HERE
    # Pista: parecido a episodio_sarsa, con dos diferencias.
    #   * La acción se elige con epsilon_greedy justo antes de ejecutarla,
    #     dentro del bucle (no hace falta arrastrar a' de una iteración a la
    #     siguiente).
    #   * El target usa np.max(Q[next_state]) en vez de Q[next_state,
    #     next_action]: por eso Q-learning es off-policy.

    state, _ = env.reset()
    # action = epsilon_greedy(Q,state,epsilon,rng)
    
    while True:

        action = epsilon_greedy(Q,state,epsilon,rng)

        next_state, reward, terminated, _, _ = env.step(action)

        Q[state,action] += alpha * (reward+gamma*Q[next_state,np.argmax(Q[next_state],axis=-1).astype(int)]- Q[state,action])

        if terminated:
            break

        state = next_state
        
    # END CODE HERE
    # ============================================================


def q_learning(
    env,
    num_episodes,
    alpha,
    gamma,
    epsilon,
    q_init=0.0,
    seed=None,
    track_states=(),
):
    """Entrena un agente con Q-learning tabular.

    Args y valor devuelto: ver :func:`entrenamiento.entrena`.
    """
    return entrena(
        env,
        episodio_q_learning,
        num_episodes,
        alpha,
        gamma,
        epsilon,
        q_init=q_init,
        seed=seed,
        track_states=track_states,
    )
