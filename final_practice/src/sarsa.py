"""Semi-gradient SARSA con aproximación lineal (Práctica 3 · Sesión 1).

SARSA es on-policy: el target usa la acción ``A'`` que el agente va a
ejecutar de verdad en ``S'``, elegida con la misma política epsilon-greedy que
está siguiendo (Sutton & Barto, §10.1):

    w <- w + alpha * [R + gamma * q̂(S', A', w) - q̂(S, A, w)] * grad q̂(S, A, w)

Hay que completar dos funciones:

* ``epsilon_sarsa``: vuestro programa de exploración, el epsilon de cada
  episodio;
* ``episodio_sarsa``: un episodio completo de semi-gradient SARSA.
"""

from __future__ import annotations

from agente import epsilon_greedy, valores_q  # noqa: F401  (en los huecos)
from entrenamiento import entrena, es_exito  # noqa: F401  (es_exito, en el hueco)


def epsilon_sarsa(k, num_episodios):
    """Epsilon con el que SARSA explora en el episodio ``k``.

    Args:
        k (int): episodio actual, de 0 a ``num_episodios - 1``.
        num_episodios (int): episodios totales del entrenamiento.

    Returns:
        float: epsilon del episodio, entre 0 y 1.
    """
    # ============================================================
    # START CODE HERE

    raise NotImplementedError("Implementa epsilon_sarsa y borra esta línea.")

    # END CODE HERE
    # ============================================================


def episodio_sarsa(env, agente, alpha, gamma, epsilon, rng):
    """Ejecuta un episodio de semi-gradient SARSA actualizando ``agente.w``.

    Args:
        env: entorno de navegación.
        agente (AgenteLineal): agente; sus pesos ``agente.w`` se actualizan.
        alpha (float): tamaño de paso.
        gamma (float): factor de descuento.
        epsilon (float): probabilidad de explorar en este episodio.
        rng (np.random.Generator): generador para ``epsilon_greedy``.

    Returns:
        tuple: ``(retorno, pasos, exito)``: suma de recompensas sin descontar,
        número de pasos y si el episodio acabó en la zona objetivo (usa
        ``es_exito``).
    """
    # ============================================================
    # START CODE HERE
    # TODO: estructura de un episodio:
    #   1. obs, _ = env.reset(); tiles activos con agente.activos(obs) y
    #      primera acción con epsilon_greedy.
    #   2. Repite: env.step(accion), elige A' en S', calcula el target,
    #      actualiza agente.w y avanza S <- S', A <- A'.
    #   3. Acumula retorno y pasos, y devuelve los tres valores al terminar.
    # HINT: env.step devuelve terminated y truncated. Solo terminated significa
    #   que S' es terminal (q̂(S', ·) = 0); un episodio truncado acaba, pero
    #   S' no es terminal.

    obs, _ = env.reset() # obs = [x, y, colision, objetivo]
    activos = agente.activos(obs)
    q = valores_q(agente.w, activos)
    accion = epsilon_greedy(q, epsilon, rng)
    steps = 0
    retorno = 0

    while True:

        steps += 1

        obs2, r, terminated, truncated, _ = env.step(accion)
        activos2 = agente.activos(obs2)
        retorno += r

        q2 = valores_q(agente.w, activos2)
        accion2 = epsilon_greedy(q2, epsilon, rng)

        agente.w[accion, activos] += alpha*(
            (
                (r if terminated else r + gamma*q2[accion2]) - q[accion]

            )
        )

        activos = activos2
        accion = accion2
        q = valores_q(agente.w, activos)

        if terminated or truncated:
            break

    exito = es_exito(obs2, terminated)

    return (retorno, steps, exito)

    # END CODE HERE
    # ============================================================


def sarsa(env, agente, num_episodios, alpha, gamma, seed=None, evalua_cada=0):
    """Entrena a ``agente`` con semi-gradient SARSA.

    Args y valor devuelto: ver :func:`entrenamiento.entrena`.
    """
    return entrena(
        env,
        agente,
        episodio_sarsa,
        epsilon_sarsa,
        num_episodios,
        alpha,
        gamma,
        seed=seed,
        evalua_cada=evalua_cada,
    )
