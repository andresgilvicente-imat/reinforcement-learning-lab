"""Semi-gradient Q-learning con aproximación lineal (Práctica 3 · Sesión 1).

Q-learning es off-policy: el agente se mueve con una política epsilon-greedy,
pero el target usa la acción greedy en ``S'`` (Sutton & Barto, §6.5 y §10.1):

    w <- w + alpha * [R + gamma * max_a q̂(S', a, w) - q̂(S, A, w)] * grad q̂(S, A, w)

Hay que completar dos funciones:

* ``epsilon_q_learning``: vuestro programa de exploración para Q-learning;
* ``episodio_q_learning``: un episodio completo de semi-gradient Q-learning.
"""

from __future__ import annotations

from agente import epsilon_greedy, valores_q  # noqa: F401  (en los huecos)
from entrenamiento import (
    entrena,
    es_exito,
)  # noqa: F401  (es_exito, en el hueco)


def epsilon_q_learning(k, num_episodios):
    """Epsilon con el que Q-learning explora en el episodio ``k``.

    Args:
        k (int): episodio actual, de 0 a ``num_episodios - 1``.
        num_episodios (int): episodios totales del entrenamiento.

    Returns:
        float: epsilon del episodio, entre 0 y 1.
    """
    # ============================================================
    # START CODE HERE

    # Params
    epsilon_inicial = 0.20
    epsilon_final = 0.001

    # Steps de fase
    fase_1 = int(0.05 * num_episodios)
    fase_2 = int(0.60 * num_episodios)

    # Fase de exploración
    if k < fase_1:
        return epsilon_inicial

    # Fase de decaimiento exponencial
    elif k < fase_2:
        progreso = (k - fase_1) / max(1, fase_2 - fase_1 - 1)

        return epsilon_inicial * (epsilon_final / epsilon_inicial) ** progreso

    # Fase de explotación
    else:
        return epsilon_final

    # END CODE HERE
    # ============================================================


def episodio_q_learning(env, agente, alpha, gamma, epsilon, rng):
    """Ejecuta un episodio de semi-gradient Q-learning actualizando ``agente.w``.

    Args y valor devuelto: como en :func:`sarsa.episodio_sarsa`.
    """
    # ============================================================
    # START CODE HERE
    # TODO: como en SARSA, pero el target no depende de la acción que se
    #   ejecutará en S'. Elige la acción con epsilon_greedy al principio de
    #   cada paso y trata terminated y truncated igual que en SARSA.

    obs, _ = env.reset()  # obs = [x, y, colision, objetivo]
    activos = agente.activos(obs)
    
    steps = 0
    retorno = 0

    while True:

        q = valores_q(agente.w, activos)
        accion = epsilon_greedy(q, epsilon, rng)

        steps += 1

        obs2, r, terminated, truncated, _ = env.step(accion)
        activos2 = agente.activos(obs2)
        retorno += r

        q2 = valores_q(agente.w, activos2)
        accion2 = epsilon_greedy(q2, 0, rng) # Change if we want more speed to just make greedy

        agente.w[accion, activos] += alpha * (
            ((r if terminated else r + gamma * q2[accion2]) - q[accion])
        )

        activos = activos2

        if terminated or truncated:
            break

    exito = es_exito(obs2, terminated)

    return (retorno, steps, exito)

    # END CODE HERE
    # ============================================================


def q_learning(
    env, agente, num_episodios, alpha, gamma, seed=None, evalua_cada=0
):
    """Entrena a ``agente`` con semi-gradient Q-learning.

    Args y valor devuelto: ver :func:`entrenamiento.entrena`.
    """
    return entrena(
        env,
        agente,
        episodio_q_learning,
        epsilon_q_learning,
        num_episodios,
        alpha,
        gamma,
        seed=seed,
        evalua_cada=evalua_cada,
    )
