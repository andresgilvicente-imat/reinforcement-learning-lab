"""Bucle de entrenamiento común a SARSA y Q-learning (Práctica 2 · RL IMAT).

Este fichero **no tiene huecos** y no hay que modificarlo. Se encarga de:

* crear la tabla Q inicial (con el Q_0 que se elija),
* repetir episodios llamando al algoritmo que toca, y
* registrar lo que hace falta para el informe: retorno por episodio, pasos por
  episodio, si el episodio acabó en el objetivo y la traza de Q(s, a) en unos
  estados concretos.

Así, en ``sarsa.py`` y ``q_learning.py`` solo queda el algoritmo: la diferencia
entre los dos es exactamente la diferencia entre on-policy y off-policy, sin
código de contabilidad alrededor.
"""

from __future__ import annotations

import numpy as np


def es_exito(env, reward, terminated):
    """``True`` si el episodio ha terminado alcanzando la celda objetivo.

    Chocar con un obstáculo también termina el episodio, así que no basta con
    mirar ``terminated``: lo que distingue los dos finales es la recompensa.
    """
    return bool(terminated) and float(reward) == float(env.unwrapped.r_goal)


def entrena(
    env,
    episodio,
    num_episodes,
    alpha,
    gamma,
    epsilon,
    q_init=0.0,
    seed=None,
    track_states=(),
):
    """Entrena ``num_episodes`` episodios con el algoritmo ``episodio``.

    Args:
        env: entorno Gymnasium con ``observation_space.n`` estados.
        episodio (callable): función que ejecuta **un** episodio completo
            actualizando ``Q`` in situ y devuelve ``(retorno, pasos, exito)``.
            Es lo que implementan ``sarsa.py`` y ``q_learning.py``.
        num_episodes (int): número de episodios de entrenamiento.
        alpha (float): tasa de aprendizaje.
        gamma (float): factor de descuento.
        epsilon (float): probabilidad de explorar durante el entrenamiento.
        q_init (float): valor inicial de todas las entradas de Q.
        seed (int | None): semilla del entorno y de la exploración. Con la
            misma semilla el entrenamiento es reproducible.
        track_states (tuple[int, ...]): estados cuya fila de Q se guarda al
            final de cada episodio, para dibujar Q(s, a) frente al número de
            pasos.

    Returns:
        dict con:
            * ``Q`` (np.ndarray[nS, nA]): tabla Q aprendida.
            * ``retornos`` (np.ndarray[num_episodes]): retorno de cada episodio.
            * ``pasos`` (np.ndarray[num_episodes]): pasos de cada episodio.
            * ``exitos`` (np.ndarray[num_episodes], bool): si acabó en el objetivo.
            * ``pasos_acumulados`` (np.ndarray[num_episodes]): pasos totales al
              terminar cada episodio (eje x de las gráficas del enunciado).
            * ``track_states`` (tuple): los estados seguidos.
            * ``q_history`` (np.ndarray[num_episodes, len(track_states), nA]):
              Q(s, a) de esos estados al final de cada episodio.
    """
    nS = env.observation_space.n
    nA = env.action_space.n

    Q = np.full((nS, nA), float(q_init), dtype=float)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    track_states = tuple(int(s) for s in track_states)
    retornos = np.zeros(num_episodes)
    pasos = np.zeros(num_episodes, dtype=int)
    exitos = np.zeros(num_episodes, dtype=bool)
    q_history = np.zeros((num_episodes, len(track_states), nA))

    for k in range(num_episodes):
        retorno, n_pasos, exito = episodio(env, Q, alpha, gamma, epsilon, rng)
        retornos[k] = retorno
        pasos[k] = n_pasos
        exitos[k] = exito
        if track_states:
            q_history[k] = Q[list(track_states)]

    return {
        "Q": Q,
        "retornos": retornos,
        "pasos": pasos,
        "exitos": exitos,
        "pasos_acumulados": np.cumsum(pasos),
        "track_states": track_states,
        "q_history": q_history,
    }
