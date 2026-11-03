"""Agente con aproximación lineal de q (Práctica 3 · Sesión 1 · RL IMAT).

El agente aproxima el valor de acción con una función lineal de las
características de tile coding, con un vector de pesos por acción:

    q̂(s, a, w) = w_a · x(s)

donde ``x(s)`` es el vector binario del tile coding y ``w_a`` la fila ``a``
de ``w``.

Hay que completar las dos funciones del final: ``valores_q`` y
``epsilon_greedy``. La clase ``AgenteLineal`` ya está hecha y no hay que
modificarla.
"""

from __future__ import annotations

import numpy as np

from representacion import TileCoder


class AgenteLineal:
    """Pesos ``w`` y tile coding: todo lo que define al agente.

    Args:
        coder (TileCoder): codificación de la posición.
        num_acciones (int): número de acciones del entorno.
        q0 (float): valor inicial de ``q̂(s, a, w)`` para todo par. Como en cada
            estado hay ``num_tilings`` tiles activos, cada peso empieza en
            ``q0 / num_tilings``.

    Atributos:
        w (np.ndarray): pesos, de forma ``(num_acciones, d)``; la fila ``a`` es
            el vector ``w_a`` de la acción ``a``.
    """

    def __init__(self, coder: TileCoder, num_acciones: int = 4, q0: float = 0.0):
        self.coder = coder
        self.num_acciones = int(num_acciones)
        self.w = np.full(
            (self.num_acciones, coder.num_caracteristicas),
            float(q0) / coder.num_tilings,
        )

    def activos(self, obs) -> np.ndarray:
        """Índices de las componentes de ``x(s)`` que valen 1."""
        return self.coder.activos(obs[0], obs[1])

    def q(self, obs) -> np.ndarray:
        """``q̂(s, a, w)`` de las acciones en la observación ``obs``."""
        return valores_q(self.w, self.activos(obs))

    def accion_greedy(self, obs) -> int:
        """Acción greedy determinista; los empates van a la de menor índice.

        Es la política con la que se evalúa al agente y con la que compite en
        la leaderboard.
        """
        return int(np.argmax(self.q(obs)))

    # ---------------------------------------------------------- persistencia
    def guarda(self, ruta, **metadatos) -> None:
        """Guarda pesos y tile coding en un ``.npz`` (sin pickle).

        Los ``metadatos`` (grupo, algoritmo, episodios...) se guardan como texto
        o números.
        """
        np.savez_compressed(
            ruta,
            w=self.w,
            n_tiles=np.array([self.coder.nx, self.coder.ny]),
            num_tilings=np.array(self.coder.num_tilings),
            **{k: np.array(v) for k, v in metadatos.items()},
        )

    @classmethod
    def carga(cls, ruta) -> AgenteLineal:
        """Reconstruye un agente a partir de un ``.npz`` de :meth:`guarda`."""
        with np.load(ruta, allow_pickle=False) as datos:
            coder = TileCoder(
                n_tiles=tuple(int(n) for n in datos["n_tiles"]),
                num_tilings=int(datos["num_tilings"]),
            )
            w = np.array(datos["w"], dtype=float)
        if w.ndim != 2 or w.shape[1] != coder.num_caracteristicas:
            raise ValueError(
                f"Los pesos tienen forma {w.shape} y el tile coding necesita "
                f"(num_acciones, {coder.num_caracteristicas})."
            )
        agente = cls(coder, num_acciones=w.shape[0])
        agente.w = w
        return agente


def valores_q(w, activos):
    """``q̂(s, a, w)`` para todas las acciones a partir de los tiles activos.

    Args:
        w (np.ndarray): pesos, de forma ``(num_acciones, d)``.
        activos (np.ndarray): índices de las componentes de ``x(s)`` a 1.

    Returns:
        np.ndarray: vector de tamaño ``num_acciones`` con ``q̂(s, a, w)``.
    """
    # ============================================================
    # START CODE HERE

    return w[:, activos].sum(axis=1)

    # END CODE HERE
    # ============================================================


def epsilon_greedy(q, epsilon, rng):
    """Elige una acción epsilon-greedy a partir de los valores ``q``.

    Con probabilidad ``epsilon`` devuelve una acción uniforme y con
    probabilidad ``1 - epsilon`` la de mayor valor (los empates se rompen al
    azar).

    Args:
        q (np.ndarray): ``q̂(s, a, w)`` de cada acción.
        epsilon (float): probabilidad de explorar.
        rng (np.random.Generator): generador del entrenamiento. Usa
            ``rng.random()``, ``rng.integers(...)`` o ``rng.choice(...)`` y no
            ``np.random``: así el entrenamiento es reproducible.

    Returns:
        int: acción elegida.
    """
    # ============================================================
    # START CODE HERE

    num_actions = q.shape[0]
    random_num = rng.random()

    if random_num < epsilon:
        return int(rng.choice([i for i in range(num_actions)]))

    return int(np.argmax(q, axis=0))

    # END CODE HERE
    # ============================================================