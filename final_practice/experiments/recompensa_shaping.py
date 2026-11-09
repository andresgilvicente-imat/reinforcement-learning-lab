"""Recompensa con *potential-based shaping* (Ng, Harada y Russell, 1999).

    r'(s, a, s') = r(s, a, s') + γ·Φ(s') − Φ(s),     Φ(terminal) = 0

con ``r`` la recompensa base de la práctica (+100 llegar, −100 chocar, −1 paso)
y ``Φ(s) = −(distancia geodésica de s a la zona objetivo, en pasos)``.

Por qué no cambia la política óptima: a lo largo de cualquier episodio el
término de shaping es telescópico,

    Σ_t γ^t (γ Φ(s_{t+1}) − Φ(s_t)) = γ^T Φ(s_T) − Φ(s_0) = −Φ(s_0),

porque el estado final es terminal (Φ = 0). Todas las políticas ven su retorno
desplazado por la MISMA constante −Φ(s_0), así que el orden entre políticas (y
la óptima) no cambia. Lo que sí cambia es la densidad de la señal: cada paso
que acerca al objetivo recibe ≈ 0 en vez de −1 y cada paso que aleja ≈ −2,
de modo que el agente no tiene que propagar la recompensa final hacia atrás
episodio a episodio.

Φ se calcula una vez, al importar el módulo, con un BFS 4-conexo sobre una
rejilla de ``H`` metros del espacio de configuraciones (centros del agente que
no chocan), usando la geometría de ``env``. Que Φ sea aproximado no afecta a la
invariancia: el resultado de Ng et al. vale para CUALQUIER función Φ.

Este fichero vive en ``experiments/`` para poder compararlo con la recompensa
base sin tocar ``src/recompensa.py``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from env import AGENT_STEP, HEIGHT, WIDTH, in_target, is_collision

GAMMA = 0.99  # el mismo γ que main.py (GAMMA); el shaping lo necesita
H = 0.05  # resolución de la rejilla de distancias (m)
R_LLEGAR, R_CHOCAR, R_PASO = 100.0, -100.0, -1.0


def _mapa_distancias(h=H):
    nx, ny = int(round(WIDTH / h)), int(round(HEIGHT / h))
    xs = (np.arange(nx) + 0.5) * h
    ys = (np.arange(ny) + 0.5) * h
    libre = np.array([[not is_collision(x, y) for y in ys] for x in xs])
    objetivo = np.array([[in_target(x, y) for y in ys] for x in xs])
    dist = np.full((nx, ny), np.inf)
    cola = deque()
    for i, j in zip(*np.nonzero(objetivo & libre)):
        dist[i, j] = 0.0
        cola.append((i, j))
    while cola:
        i, j = cola.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and libre[a, b] and dist[a, b] == np.inf:
                dist[a, b] = dist[i, j] + h
                cola.append((a, b))
    # Celdas del borde cuyo centro choca pero que el agente puede pisar con el
    # suyo: se rellenan con el vecino finito más cercano.
    for _ in range(3):
        inf = ~np.isfinite(dist)
        if not inf.any():
            break
        relleno = dist.copy()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            vecino = np.roll(dist, d, axis=(0, 1)) + h
            relleno = np.where(inf, np.minimum(relleno, vecino), relleno)
        dist = relleno
    return dist / AGENT_STEP  # en pasos del agente


_DIST = _mapa_distancias()


def potencial(x, y):
    i = min(max(int(x / H), 0), _DIST.shape[0] - 1)
    j = min(max(int(y / H), 0), _DIST.shape[1] - 1)
    d = _DIST[i, j]
    return -d if np.isfinite(d) else -float(np.nanmax(_DIST[np.isfinite(_DIST)]))


def recompensa(pos, nueva_pos, colision, objetivo):
    if objetivo:
        base, phi_sig = R_LLEGAR, 0.0  # terminal: Φ = 0
    elif colision:
        base, phi_sig = R_CHOCAR, 0.0  # terminal: Φ = 0
    else:
        base, phi_sig = R_PASO, potencial(*nueva_pos)
    return base + GAMMA * phi_sig - potencial(*pos)
