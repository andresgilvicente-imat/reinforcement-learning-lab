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
    #
    # Recompensa base: +100 al llegar, -100 al chocar, -1 por paso.
    # Sobre ella, *potential-based shaping* (Ng, Harada y Russell, 1999):
    #
    #     r'(s, a, s') = r(s, a, s') + γ·Φ(s') − Φ(s),   Φ(terminal) = 0,
    #
    # con Φ(s) = −(distancia geodésica de s a la zona objetivo, en pasos).
    # En cualquier episodio los términos de shaping se cancelan en cadena:
    #     Σ_t γ^t (γΦ(s_{t+1}) − Φ(s_t)) = γ^T Φ(s_T) − Φ(s_0) = −Φ(s_0),
    # así que TODAS las políticas ven su retorno desplazado por la misma
    # constante y la política óptima no cambia. Lo que cambia es la densidad
    # de la señal: un paso que acerca al objetivo recibe ≈ 0 y uno que aleja
    # ≈ −2, en lugar de −1 en ambos casos.
    #
    # Φ se calcula una sola vez (primera llamada) con un BFS 4-conexo sobre una
    # rejilla de 5 cm de las posiciones libres del recinto, usando la geometría
    # de env, y se guarda en la propia función.
    import numpy as np
    from collections import deque

    from env import AGENT_STEP, HEIGHT, WIDTH, in_target, is_collision

    GAMMA = 0.99  # el γ de main.py (GAMMA); el shaping debe usar el mismo
    H = 0.05  # resolución de la rejilla de distancias (m)

    if not hasattr(recompensa, "_dist"):
        nx, ny = int(round(WIDTH / H)), int(round(HEIGHT / H))
        xs = (np.arange(nx) + 0.5) * H
        ys = (np.arange(ny) + 0.5) * H
        libre = np.array([[not is_collision(x, y) for y in ys] for x in xs])
        meta = np.array([[in_target(x, y) for y in ys] for x in xs])
        dist = np.full((nx, ny), np.inf)
        cola = deque()
        for i, j in zip(*np.nonzero(meta & libre)):
            dist[i, j] = 0.0
            cola.append((i, j))
        while cola:
            i, j = cola.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and libre[a, b] and dist[a, b] == np.inf:
                    dist[a, b] = dist[i, j] + H
                    cola.append((a, b))
        # Celdas del borde cuyo centro choca pero que el agente puede pisar:
        # se rellenan con el vecino finito más cercano.
        for _ in range(3):
            inf = ~np.isfinite(dist)
            if not inf.any():
                break
            relleno = dist.copy()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                vecino = np.roll(dist, d, axis=(0, 1)) + H
                relleno = np.where(inf, np.minimum(relleno, vecino), relleno)
            dist = relleno
        recompensa._dist = dist / AGENT_STEP  # en pasos del agente

    dist = recompensa._dist

    def potencial(x, y):
        i = min(max(int(x / H), 0), dist.shape[0] - 1)
        j = min(max(int(y / H), 0), dist.shape[1] - 1)
        d = dist[i, j]
        return -d if np.isfinite(d) else -float(np.nanmax(dist[np.isfinite(dist)]))

    if objetivo:
        base, phi_sig = 100.0, 0.0  # terminal: Φ = 0
    elif colision:
        base, phi_sig = -100.0, 0.0  # terminal: Φ = 0
    else:
        base, phi_sig = -1.0, potencial(*nueva_pos)
    return base + GAMMA * phi_sig - potencial(*pos)

    # END CODE HERE
    # ============================================================
