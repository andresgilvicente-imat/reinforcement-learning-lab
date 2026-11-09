"""Recompensa original, sin shaping (la que había en src/recompensa.py en e1-e5).

Se conserva aquí para poder reproducir los experimentos antiguos
(``barrido.py --recompensa base``) aunque src/recompensa.py ya tenga el shaping.
"""

from __future__ import annotations


def recompensa(pos, nueva_pos, colision, objetivo):
    if objetivo:
        return 100.0
    if colision:
        return -100.0
    return -1.0
