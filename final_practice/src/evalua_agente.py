"""Comprueba un agente guardado antes de enviarlo a la leaderboard.

Carga el ``.npz`` igual que lo cargarán los profesores (solo pesos y diseño
del tile coding: vuestro código no se ejecuta) y lo evalúa con su política
greedy en posiciones aleatorias. La leaderboard hace exactamente lo mismo, con
otras posiciones que no se publican.

    python evalua_agente.py agentes/G7_sarsa.npz
    python evalua_agente.py agentes/G7_sarsa.npz --n 1000 --seed 5

Este fichero NO debe modificarse durante la práctica.
"""

from __future__ import annotations

import argparse

import numpy as np

from agente import AgenteLineal
from evaluacion import SEMILLA_VALIDACION, evalua, posiciones_aleatorias


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa un agente guardado con su política greedy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fichero", help="Fichero .npz guardado con main.py --guarda")
    parser.add_argument("--n", type=int, default=100, help="Número de posiciones.")
    parser.add_argument("--seed", type=int, default=SEMILLA_VALIDACION)
    args = parser.parse_args()

    agente = AgenteLineal.carga(args.fichero)
    with np.load(args.fichero, allow_pickle=False) as datos:
        metadatos = {
            k: datos[k].item()
            for k in datos.files
            if k not in ("w", "n_tiles", "num_tilings")
        }
    print(f"{args.fichero}: {agente.coder}")
    print(f"Metadatos: {metadatos}")

    resultado = evalua(agente, posiciones_aleatorias(args.n, seed=args.seed))
    finales = dict(zip(*np.unique(resultado["final"], return_counts=True)))
    print(
        f"Llega {resultado['llegadas']} de {args.n} "
        f"({resultado['tasa_exito']:.1%}) · pasos medios de las llegadas "
        f"{resultado['pasos_medios']:.2f}"
    )
    print(f"Finales: {finales}")


if __name__ == "__main__":
    main()
