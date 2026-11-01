"""Bucle de entrenamiento común a SARSA y Q-learning (Práctica 3 · Sesión 1).

Este fichero **no tiene huecos** y no hay que modificarlo. Se encarga de:

* pedir a vuestro programa de exploración el ``epsilon`` de cada episodio,
* repetir episodios llamando al algoritmo que toca, y
* registrar lo que hace falta para analizar el aprendizaje: retorno, pasos y
  éxito de cada episodio, el ``epsilon`` usado y, cada ``evalua_cada``
  episodios, cómo lo hace la política greedy en un conjunto fijo de posiciones
  de validación (lo mismo que mide la leaderboard).

Así, en ``sarsa.py`` y ``q_learning.py`` solo queda el algoritmo y la
exploración.

**Toda la aleatoriedad sale de la semilla** (``seed``): el generador ``rng``
que reciben vuestras funciones, el del entorno, el de ``env.action_space`` y los
globales de ``numpy`` y ``random``. Con el mismo código, los mismos
hiperparámetros y la misma semilla, los pesos salen idénticos: así comprobamos
los agentes de la leaderboard.
"""

from __future__ import annotations

import random

import numpy as np

from evaluacion import evalua, posiciones_aleatorias

NUM_POSICIONES_VALIDACION = 100


def es_exito(obs, terminated) -> bool:
    """``True`` si el episodio ha terminado en la zona objetivo.

    Chocar también termina el episodio: lo que distingue los dos finales es
    la última componente de la observación cruda, ``objetivo``.
    """
    return bool(terminated) and bool(obs[3])


def _comprueba_pesos(agente, k):
    if not np.isfinite(agente.w).all():
        raise FloatingPointError(
            f"Los pesos han divergido (inf o nan) en el episodio {k}."
        )


def fija_semillas(env, seed):
    """Fija todas las fuentes de aleatoriedad y devuelve el ``rng`` del agente."""
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return np.random.default_rng(seed)


def entrena(
    env,
    agente,
    episodio,
    programa_epsilon,
    num_episodios,
    alpha,
    gamma,
    seed=None,
    evalua_cada=0,
):
    """Entrena a ``agente`` durante ``num_episodios`` episodios.

    Args:
        env: entorno de navegación.
        agente (AgenteLineal): agente cuyos pesos ``agente.w`` se actualizan.
        episodio (callable): ``episodio(env, agente, alpha, gamma, epsilon,
            rng) -> (retorno, pasos, exito)``. Ejecuta **un** episodio
            completo actualizando ``agente.w``.
        programa_epsilon (callable): ``programa_epsilon(k, num_episodios) ->
            epsilon``, el epsilon del episodio ``k`` (de 0 a
            ``num_episodios - 1``).
        num_episodios (int): episodios de entrenamiento.
        alpha (float): tamaño de paso de la actualización semi-gradiente.
        gamma (float): factor de descuento.
        seed (int | None): semilla de todo el entrenamiento (ver
            :func:`fija_semillas`). Con ``None`` no es reproducible.
        evalua_cada (int): cada cuántos episodios se evalúa la política greedy
            en las posiciones de validación (0 = nunca).

    Returns:
        dict con ``retornos``, ``pasos``, ``exitos`` y ``epsilons`` (un valor
        por episodio), ``pasos_acumulados`` y, si ``evalua_cada > 0``,
        ``validacion``: lista de ``(episodios, tasa_exito, pasos_medios)``.
    """
    rng = fija_semillas(env, seed)
    posiciones = posiciones_aleatorias(NUM_POSICIONES_VALIDACION)

    retornos = np.zeros(num_episodios)
    pasos = np.zeros(num_episodios, dtype=int)
    exitos = np.zeros(num_episodios, dtype=bool)
    epsilons = np.zeros(num_episodios)
    validacion = []

    for k in range(num_episodios):
        epsilon = float(programa_epsilon(k, num_episodios))
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon = {epsilon} en el episodio {k}: fuera de [0, 1]")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                retorno, n_pasos, exito = episodio(
                    env, agente, alpha, gamma, epsilon, rng
                )
        except Exception:
            # Si los pesos ya no son finitos, el fallo es la divergencia.
            _comprueba_pesos(agente, k)
            raise
        _comprueba_pesos(agente, k)
        retornos[k] = retorno
        pasos[k] = n_pasos
        exitos[k] = exito
        epsilons[k] = epsilon
        if evalua_cada and (k + 1) % evalua_cada == 0:
            resultado = evalua(agente, posiciones)
            validacion.append(
                (k + 1, resultado["tasa_exito"], resultado["pasos_medios"])
            )

    return {
        "retornos": retornos,
        "pasos": pasos,
        "exitos": exitos,
        "epsilons": epsilons,
        "pasos_acumulados": np.cumsum(pasos),
        "validacion": validacion,
    }
