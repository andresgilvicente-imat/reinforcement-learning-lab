"""Diseño de la recompensa (Práctica 3 · Sesión 1 · RL IMAT).

El entorno llama a ``recompensa`` en cada paso, después de mover al agente, y
devuelve lo que esta función calcule como ``reward`` de ``env.step``. Lo que
decidáis aquí define la tarea que el agente aprende de verdad: comprobad que
maximizar el retorno coincide con lo que queréis (llegar a la zona objetivo,
sin chocar y por el camino más corto).

Si necesitáis la geometría del recinto, importadla de ``env`` (``TARGET_AREA``,
``OBSTACLES``, ``WIDTH``, ``HEIGHT``...).
"""

from __future__ import annotations


def recompensa(pos, nueva_pos, colision, objetivo):
    """Recompensa de una transición.

    Args:
        pos (tuple[float, float]): posición ``(x, y)`` antes de moverse.
        nueva_pos (tuple[float, float]): posición a la que intentaba moverse.
            Si ha chocado, el agente no llega a ocuparla.
        colision (bool): si el movimiento ha chocado con una pared u obstáculo.
        objetivo (bool): si el agente ha entrado en la zona objetivo.

    Returns:
        float: recompensa de la transición.
    """
    # ============================================================
    # START CODE HERE
    if objetivo:
        return 100.0

    if colision:
        return -100.0

    return -1.0

    # END CODE HERE
    # ============================================================
