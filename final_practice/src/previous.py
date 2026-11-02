"""Exploración sistemática del área de influencia de una muestra.

Barre combinaciones de ``n_tiles``, ``num_tilings`` y tipo de desplazamiento,
y agrupa los resultados en figuras para ver el efecto de cada parámetro:

  1. rejilla_<desp>.png      filas = n_tiles, columnas = num_tilings
                             (una figura por tipo de desplazamiento)
  2. comparacion_n<N>.png    filas = asimétrico / uniforme, columnas = num_tilings
                             (una figura por n_tiles; equivale a la Figura 9.11)
  3. area_vs_parametros.png  área de influencia (m²) frente a los parámetros

Uso:
    python explorar.py
    python explorar.py --punto 5 7.5 --salida figuras
    python explorar.py --n_tiles 2 4 8 16 --num_tilings 1 4 16 --resolucion 0.1
    python explorar.py --con_tiles          # superpone los tiles activos

Usa ``fraccion_compartida`` de influencia.py, que no se modifica.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from env import HEIGHT, WIDTH, draw_environment
from influencia import DESPLAZAMIENTOS, fraccion_compartida
from representacion import TileCoder


def calcula(n_tiles_lista, tilings_lista, punto, resolucion):
    """Calcula la fracción compartida para todas las combinaciones.

    Returns:
        dict: ``(desp, n, k) -> (coder, fraccion, area)``.
    """
    resultados = {}
    total = len(DESPLAZAMIENTOS) * len(n_tiles_lista) * len(tilings_lista)
    hecho = 0
    for desp, valor in DESPLAZAMIENTOS.items():
        for n in n_tiles_lista:
            for k in tilings_lista:
                coder = TileCoder(n_tiles=(n, n), num_tilings=k, displacement=valor)
                _, _, fr = fraccion_compartida(coder, punto, resolucion)
                area = (fr > 0).mean() * WIDTH * HEIGHT
                resultados[(desp, n, k)] = (coder, fr, area)
                hecho += 1
                print(f"[{hecho}/{total}] {desp:10s} n_tiles={n:<3d} "
                      f"tilings={k:<3d} área={area:6.1f} m²")
    return resultados


def pinta_celda(ax, coder, fr, area, punto, con_tiles, titulo=None):
    """Dibuja una celda: mapa de influencia (+ tiles activos opcionales)."""
    im = ax.imshow(fr, origin="lower", extent=(0, WIDTH, 0, HEIGHT),
                   cmap="magma", vmin=0, vmax=1)
    draw_environment(ax)
    if con_tiles:
        colores = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, (x0, y0) in enumerate(coder.esquinas(*punto)):
            ax.add_patch(Rectangle((x0, y0), coder.tile_width, coder.tile_height,
                                   fill=False, edgecolor=colores[i % 10],
                                   linewidth=0.8, alpha=0.8))
    ax.plot(*punto, "o", color="cyan", markersize=4)
    ax.set_title(titulo or f"{area:.1f} m²", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def guarda(fig, ruta):
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"Guardada: {ruta}")


def fig_rejilla(res, desp, n_lista, k_lista, punto, con_tiles, ruta):
    """Filas = n_tiles, columnas = num_tilings, para un desplazamiento."""
    ancho = 2.6 * len(k_lista) * WIDTH / HEIGHT * 0.8
    fig, axes = plt.subplots(len(n_lista), len(k_lista), squeeze=False,
                             figsize=(max(ancho, 6), 2.6 * len(n_lista)))
    im = None
    for i, n in enumerate(n_lista):
        for j, k in enumerate(k_lista):
            coder, fr, area = res[(desp, n, k)]
            im = pinta_celda(axes[i][j], coder, fr, area, punto, con_tiles)
            if i == 0:
                axes[i][j].set_title(f"{k} tilings\n{area:.1f} m²", fontsize=9)
            if j == 0:
                axes[i][j].set_ylabel(f"{n}x{n} tiles", fontsize=10)
    fig.suptitle(f"Área de influencia en {punto} – desplazamiento {desp}\n"
                 "(filas: tiles por dimensión · columnas: nº de tilings)")
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    cax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cax, label="fracción de tiles compartidos")
    guarda(fig, ruta)


def fig_comparacion(res, n, k_lista, punto, con_tiles, ruta):
    """Asimétrico vs uniforme para un n_tiles dado (cf. Figura 9.11)."""
    desps = list(DESPLAZAMIENTOS)
    fig, axes = plt.subplots(len(desps), len(k_lista), squeeze=False,
                             figsize=(2.8 * len(k_lista), 2.9 * len(desps)))
    im = None
    for i, desp in enumerate(desps):
        for j, k in enumerate(k_lista):
            coder, fr, area = res[(desp, n, k)]
            im = pinta_celda(axes[i][j], coder, fr, area, punto, con_tiles)
            if i == 0:
                axes[i][j].set_title(f"{k} tilings\n{area:.1f} m²", fontsize=9)
            if j == 0:
                axes[i][j].set_ylabel(desp, fontsize=10)
    fig.suptitle(f"Asimétrico vs uniforme – {n}x{n} tiles, punto {punto}")
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    cax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cax, label="fracción de tiles compartidos")
    guarda(fig, ruta)


def fig_areas(res, n_lista, k_lista, ruta):
    """Área de influencia frente a num_tilings y a n_tiles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    estilos = {"asimetrico": "-o", "uniforme": "--s"}
    colores = plt.cm.viridis(np.linspace(0, 0.9, len(n_lista)))
    for desp, est in estilos.items():
        for c, n in zip(colores, n_lista):
            axes[0].plot(k_lista, [res[(desp, n, k)][2] for k in k_lista],
                         est, color=c, label=f"{desp}, {n}x{n}")
    axes[0].set_xlabel("num_tilings")
    axes[0].set_ylabel("área con fracción > 0 (m²)")
    axes[0].set_title("Área vs nº de tilings")
    axes[0].set_xscale("log", base=2)
    colores = plt.cm.plasma(np.linspace(0, 0.9, len(k_lista)))
    for desp, est in estilos.items():
        for c, k in zip(colores, k_lista):
            axes[1].plot(n_lista, [res[(desp, n, k)][2] for n in n_lista],
                         est, color=c, label=f"{desp}, {k} tilings")
    axes[1].set_xlabel("tiles por dimensión")
    axes[1].set_title("Área vs resolución de la rejilla")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    guarda(fig, ruta)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_tiles", type=int, nargs="+", default=[2, 4, 6, 10, 20],
                   help="valores de tiles por dimensión (rejilla cuadrada)")
    p.add_argument("--num_tilings", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--punto", type=float, nargs=2, default=(5.0, 7.5),
                   metavar=("X", "Y"))
    p.add_argument("--resolucion", type=float, default=0.1,
                   help="paso de la malla de evaluación en m (menor = más lento)")
    p.add_argument("--con_tiles", action="store_true",
                   help="superpone los tiles activos en cada celda")
    p.add_argument("--salida", type=str, default="figuras")
    p.add_argument("--mostrar", action="store_true",
                   help="muestra las figuras al terminar (abre ventanas)")
    args = p.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    punto = tuple(args.punto)
    n_lista, k_lista = sorted(args.n_tiles), sorted(args.num_tilings)

    res = calcula(n_lista, k_lista, punto, args.resolucion)

    for desp in DESPLAZAMIENTOS:
        fig_rejilla(res, desp, n_lista, k_lista, punto, args.con_tiles,
                    os.path.join(args.salida, f"rejilla_{desp}.png"))
    for n in n_lista:
        fig_comparacion(res, n, k_lista, punto, args.con_tiles,
                        os.path.join(args.salida, f"comparacion_n{n}.png"))
    fig_areas(res, n_lista, k_lista,
              os.path.join(args.salida, "area_vs_parametros.png"))

    if args.mostrar:
        import subprocess, sys
        print("Abre la carpeta", args.salida, "para ver las figuras.")


if __name__ == "__main__":
    main()