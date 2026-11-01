"""Área de influencia de una muestra de experiencia (trabajo previo, sesión 1).

Con aproximación lineal y tile coding, una actualización en el estado ``s``
mueve los pesos de sus tiles activos, y con ellos el valor de todos los
estados ``s'`` que comparten alguno de esos tiles. El cambio en
``q̂(s', a, w)`` es proporcional al número de tiles que comparten:

    Δq̂(s', a, w) ∝ x(s) · x(s') / num_tilings

Este script dibuja esa fracción sobre el recinto para un punto ``s`` y un
diseño de tile coding, junto con los tiles activos de cada tiling:

    python influencia.py --n_tiles 10 10 --num_tilings 8 --punto 5 7.5
    python influencia.py --n_tiles 4 4 --num_tilings 1 --punto 5 7.5 --guarda f.png
    python influencia.py --num_tilings 16 --desplazamiento uniforme

Este fichero NO debe modificarse durante la práctica.
"""

from __future__ import annotations

import argparse

import numpy as np

from env import HEIGHT, WIDTH, draw_environment
from representacion import DISPLACEMENT, TileCoder

DESPLAZAMIENTOS = {"asimetrico": DISPLACEMENT, "uniforme": (1, 1)}


def fraccion_compartida(coder, punto, resolucion=0.05):
    """Fracción de tiles activos que cada punto del recinto comparte con ``punto``.

    Returns:
        tuple: ``(xs, ys, fraccion)`` con ``fraccion[i, j]`` la de ``(xs[j], ys[i])``.
    """
    activos = set(coder.activos(*punto).tolist())
    xs = np.arange(resolucion / 2, WIDTH, resolucion)
    ys = np.arange(resolucion / 2, HEIGHT, resolucion)
    fraccion = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            comunes = activos.intersection(coder.activos(x, y).tolist())
            fraccion[i, j] = len(comunes) / coder.num_tilings
    return xs, ys, fraccion


def dibuja(coder, punto, ruta=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _, _, fraccion = fraccion_compartida(coder, punto)
    fig, (ax_tiles, ax_infl) = plt.subplots(1, 2, figsize=(13, 6))

    draw_environment(ax_tiles)
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, (x0, y0) in enumerate(coder.esquinas(*punto)):
        ax_tiles.add_patch(
            Rectangle(
                (x0, y0),
                coder.tile_width,
                coder.tile_height,
                fill=False,
                edgecolor=colores[k % 10],
                linewidth=1.2,
            )
        )
    ax_tiles.plot(*punto, "o", color="orange", markersize=8)
    ax_tiles.set_title(f"Tiles activos en {punto} ({coder.num_tilings} tilings)")

    im = ax_infl.imshow(
        fraccion,
        origin="lower",
        extent=(0, WIDTH, 0, HEIGHT),
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    draw_environment(ax_infl)
    ax_infl.plot(*punto, "o", color="cyan", markersize=6)
    fig.colorbar(im, ax=ax_infl, label="fracción de tiles compartidos")
    area = (fraccion > 0).mean() * WIDTH * HEIGHT
    ax_infl.set_title(f"Área de influencia: {area:.1f} m²")
    fig.suptitle(repr(coder))
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
        print(f"Figura guardada en {ruta}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Área de influencia de una muestra con tile coding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_tiles", type=int, nargs=2, default=(10, 10), metavar=("NX", "NY")
    )
    parser.add_argument("--num_tilings", type=int, default=8)
    parser.add_argument(
        "--punto", type=float, nargs=2, default=(5.0, 7.5), metavar=("X", "Y")
    )
    parser.add_argument(
        "--desplazamiento",
        choices=list(DESPLAZAMIENTOS),
        default="asimetrico",
        help="asimetrico = (1, 3), el que usa el agente; uniforme = (1, 1).",
    )
    parser.add_argument("--guarda", type=str, default=None, metavar="FICHERO")
    args = parser.parse_args()

    coder = TileCoder(
        n_tiles=args.n_tiles,
        num_tilings=args.num_tilings,
        displacement=DESPLAZAMIENTOS[args.desplazamiento],
    )
    print(f"{coder}: cada tile mide {coder.tile_width:g} x {coder.tile_height:g} m")
    dibuja(coder, tuple(args.punto), args.guarda)


if __name__ == "__main__":
    main()
