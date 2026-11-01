"""Tile coding de la posición del agente (Práctica 3 · Sesión 1 · RL IMAT).

Convierte la posición continua ``(x, y)`` en el vector de características
binario ``x(s)`` de Sutton & Barto §9.5.4: ``num_tilings`` rejillas (tilings)
de ``nx`` x ``ny`` tiles que cubren el recinto, cada una desplazada una
fracción de tile respecto a la anterior. En cada tiling hay exactamente un
tile activo, así que ``x(s)`` tiene ``num_tilings`` unos y el resto ceros.

Como ``x(s)`` es binario, no se guarda entero: ``activos(x, y)`` devuelve los
índices de sus componentes a 1.

Los desplazamientos son los que recomiendan Sutton y Barto, y los mismos que
usa su biblioteca ``tiles3``: el tiling ``i`` se desplaza ``(1, 3) * i /
num_tilings`` de tile en ``(x, y)``. Con desplazamientos uniformes (todos los
tilings desplazados en la diagonal) la generalización tiene forma de diagonal;
con estos, es más parecida en todas las direcciones.

A diferencia de ``tiles3``, aquí no hay tabla hash: el índice de cada tile se
calcula directamente, así que el mismo ``(nx, ny, num_tilings)`` da siempre
los mismos índices, en cualquier ordenador. Por eso basta con los pesos ``w``
y esos tres números para reconstruir un agente.

Este fichero NO debe modificarse durante la práctica.
"""

from __future__ import annotations

import numpy as np

from env import HEIGHT, WIDTH

# Desplazamiento de cada tiling, en fracciones de tile (Sutton y Barto, §9.5.4)
DISPLACEMENT = (1, 3)


class TileCoder:
    """Tile coding de un espacio rectangular de dos dimensiones.

    Args:
        n_tiles (tuple[int, int]): tiles por dimensión ``(nx, ny)`` en cada
            tiling. Cada tile mide ``WIDTH / nx`` x ``HEIGHT / ny`` metros.
        num_tilings (int): número de tilings superpuestos.
        dims (tuple[float, float]): tamaño del espacio ``(ancho, alto)``.
        displacement (tuple[int, int]): desplazamiento entre tilings
            consecutivos, en unidades de ``1 / num_tilings`` de tile. El agente
            usa siempre ``DISPLACEMENT``; ``(1, 1)`` son los desplazamientos
            uniformes de la figura 9.11 de Sutton y Barto.
    """

    def __init__(
        self,
        n_tiles=(10, 10),
        num_tilings=8,
        dims=(WIDTH, HEIGHT),
        displacement=DISPLACEMENT,
    ):
        self.nx, self.ny = (int(n) for n in n_tiles)
        self.num_tilings = int(num_tilings)
        self.width, self.height = (float(d) for d in dims)
        if self.nx < 1 or self.ny < 1 or self.num_tilings < 1:
            raise ValueError("n_tiles y num_tilings deben ser enteros positivos.")
        self.tile_width = self.width / self.nx
        self.tile_height = self.height / self.ny
        # Al desplazar, cada tiling necesita un tile más por dimensión.
        self._cols = self.nx + 1
        self._rows = self.ny + 1
        self._tiling = np.arange(self.num_tilings)
        n = self.num_tilings
        self._offset_x = (displacement[0] * self._tiling) % n
        self._offset_y = (displacement[1] * self._tiling) % n

    @property
    def num_caracteristicas(self) -> int:
        """Dimensión de ``x(s)``: número total de tiles de todos los tilings."""
        return self.num_tilings * self._cols * self._rows

    def activos(self, x, y) -> np.ndarray:
        """Índices de los ``num_tilings`` tiles activos en ``(x, y)``.

        El índice ``k`` de la salida es el tile activo del tiling ``k``.
        """
        n = self.num_tilings
        # Posición en unidades de 1/n de tile, como en tiles3
        qx = int(np.floor(x / self.tile_width * n))
        qy = int(np.floor(y / self.tile_height * n))
        col = (qx + self._offset_x) // n
        row = (qy + self._offset_y) // n
        return (self._tiling * self._cols + col) * self._rows + row

    def esquinas(self, x, y) -> np.ndarray:
        """Esquina inferior izquierda, en metros, del tile activo de cada tiling.

        Solo sirve para dibujar los tiles activos.
        """
        n = self.num_tilings
        qx = np.floor(x / self.tile_width * n)
        qy = np.floor(y / self.tile_height * n)
        col = (qx + self._offset_x) // n
        row = (qy + self._offset_y) // n
        x0 = (col * n - self._offset_x) / n * self.tile_width
        y0 = (row * n - self._offset_y) / n * self.tile_height
        return np.stack([x0, y0], axis=1)

    def __repr__(self):
        return (
            f"TileCoder(n_tiles=({self.nx}, {self.ny}), "
            f"num_tilings={self.num_tilings}, d={self.num_caracteristicas})"
        )


if __name__ == "__main__":
    coder = TileCoder(n_tiles=(10, 10), num_tilings=8)
    print(coder)
    for punto in ((5.0, 3.0), (5.1, 3.0), (3.0, 9.0)):
        print(f"{punto} -> {coder.activos(*punto)}")
