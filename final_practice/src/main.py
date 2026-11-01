"""Script principal de la Práctica 3, sesión 1 (Aprendizaje por Refuerzo · IMAT).

Entrena un agente lineal con tile coding mediante semi-gradient SARSA y/o
semi-gradient Q-learning en el entorno de navegación, lo evalúa con su política
greedy en las posiciones de validación y, si se pide, guarda sus pesos para la
leaderboard.

Uso básico:

    python main.py --algoritmo sarsa --alpha <alpha>
    python main.py --algoritmo ambos --alpha <alpha> --episodios 5000

Opciones útiles:

    --n_tiles 10 10 --num_tilings 8     # diseño del tile coding
    --gamma 0.99 --q0 0                 # descuento y valor inicial de q̂
    --graficas                          # curvas de aprendizaje y validación
    --mapa                              # max_a q̂(s, a) y política greedy
    --render 3                          # ver 3 episodios de la política greedy
    --guarda agentes/G7 --grupo G7      # pesos para la leaderboard

La leaderboard solo admite agentes entrenados con PRESUPUESTO_EPISODIOS episodios
como máximo (el valor por defecto de ``--episodios``).

``alpha`` no tiene valor por defecto: elegidlo vosotros. El epsilon de cada
episodio lo deciden ``epsilon_sarsa`` y ``epsilon_q_learning``.
No hace falta editar este fichero: todo se cambia desde la línea de comandos.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from agente import AgenteLineal
from env import ACTION_NAMES, NavigationEnv
from evaluacion import evalua, posiciones_aleatorias
from q_learning import q_learning
from representacion import TileCoder
from sarsa import sarsa

# --- Hiperparámetros por defecto -------------------------------------------
# Episodios máximos de un agente de la leaderboard
PRESUPUESTO_EPISODIOS = 2000
NUM_EPISODIOS = PRESUPUESTO_EPISODIOS
GAMMA = 0.99
N_TILES = (10, 10)
NUM_TILINGS = 8
Q0 = 0.0
SEED = 0
EVALUA_CADA = 100
# Episodios sobre los que se promedian las curvas de entrenamiento
VENTANA = 200

ALGORITMOS = {"sarsa": sarsa, "q_learning": q_learning}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Práctica 3, sesión 1: navegación con tile coding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algoritmo",
        choices=["sarsa", "q_learning", "ambos"],
        default="sarsa",
        help="Algoritmo que se entrena.",
    )
    parser.add_argument(
        "--alpha", type=float, required=True, help="Tamaño de paso (obligatorio)."
    )
    parser.add_argument(
        "--episodios",
        type=int,
        default=NUM_EPISODIOS,
        help="Episodios de entrenamiento (la leaderboard admite hasta 2000).",
    )
    parser.add_argument(
        "--gamma", type=float, default=GAMMA, help="Factor de descuento."
    )
    parser.add_argument(
        "--n_tiles",
        type=int,
        nargs=2,
        default=N_TILES,
        metavar=("NX", "NY"),
        help="Tiles por dimensión en cada tiling.",
    )
    parser.add_argument(
        "--num_tilings", type=int, default=NUM_TILINGS, help="Número de tilings."
    )
    parser.add_argument(
        "--q0", type=float, default=Q0, help="Valor inicial de q̂(s, a, w)."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Semilla de todo el entrenamiento."
    )
    parser.add_argument(
        "--evalua_cada",
        type=int,
        default=EVALUA_CADA,
        help="Cada cuántos episodios se evalúa la política greedy (0 = nunca).",
    )
    parser.add_argument(
        "--graficas", action="store_true", help="Curvas de aprendizaje."
    )
    parser.add_argument(
        "--mapa", action="store_true", help="max_a q̂ y política greedy."
    )
    parser.add_argument(
        "--render", type=int, default=0, metavar="N", help="Episodios a visualizar."
    )
    parser.add_argument(
        "--guarda",
        type=str,
        default=None,
        metavar="PREFIJO",
        help="Guarda los pesos en PREFIJO_<algoritmo>.npz.",
    )
    parser.add_argument(
        "--grupo",
        type=str,
        default="sin_grupo",
        help="Grupo (obligatorio con --guarda).",
    )
    return parser


def entrena_uno(nombre, args):
    env = NavigationEnv()
    coder = TileCoder(n_tiles=args.n_tiles, num_tilings=args.num_tilings)
    agente = AgenteLineal(coder, num_acciones=env.action_space.n, q0=args.q0)
    print(f"\n=== {nombre.upper()} · {coder} · alpha = {args.alpha} ===")
    inicio = time.time()
    historial = ALGORITMOS[nombre](
        env,
        agente,
        args.episodios,
        args.alpha,
        args.gamma,
        seed=args.seed,
        evalua_cada=args.evalua_cada,
    )
    duracion = time.time() - inicio
    env.close()

    exitos = historial["exitos"]
    print(
        f"{args.episodios} episodios, {historial['pasos_acumulados'][-1]} pasos, "
        f"{duracion:.0f} s"
    )
    print(
        f"Últimos {VENTANA} episodios: éxito {exitos[-VENTANA:].mean():.1%}, "
        f"pasos medios {historial['pasos'][-VENTANA:].mean():.1f}, "
        f"epsilon final {historial['epsilons'][-1]:.3f}"
    )
    if historial["validacion"]:
        print("Validación de la política greedy (episodios: éxito, pasos medios)")
        for episodios, tasa, pasos in historial["validacion"]:
            print(f"  {episodios:6d}: {tasa:6.1%}  {pasos:6.1f}")
    return agente, historial


def informa_evaluacion(nombre, agente):
    resultado = evalua(agente, posiciones_aleatorias(100))
    finales = dict(zip(*np.unique(resultado["final"], return_counts=True)))
    print(
        f"{nombre}: llega {resultado['llegadas']} de 100 · pasos medios "
        f"{resultado['pasos_medios']:.2f} · finales {finales}"
    )


def media_movil(valores, ventana=VENTANA):
    valores = np.asarray(valores, dtype=float)
    if len(valores) < ventana:
        return valores
    return np.convolve(valores, np.ones(ventana) / ventana, mode="valid")


def dibuja_graficas(historiales):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for nombre, h in historiales.items():
        x = np.arange(len(h["retornos"]))[VENTANA - 1 :]
        axes[0, 0].plot(x, media_movil(h["retornos"]), label=nombre)
        axes[0, 1].plot(x, media_movil(h["exitos"]), label=nombre)
        axes[1, 0].plot(h["epsilons"], label=nombre)
        if h["validacion"]:
            val = np.array(h["validacion"], dtype=float)
            axes[1, 1].plot(val[:, 0], val[:, 1], "o-", label=f"{nombre}: éxito")
    axes[0, 0].set_title(f"Retorno (media móvil de {VENTANA})")
    axes[0, 1].set_title(f"Éxito en entrenamiento (media móvil de {VENTANA})")
    axes[1, 0].set_title("Epsilon")
    axes[1, 1].set_title("Política greedy en validación: tasa de éxito")
    for ax in axes.flat:
        ax.set_xlabel("episodio")
        ax.legend()
    fig.tight_layout()


def dibuja_mapa(nombre, agente, resolucion=0.1):
    import matplotlib.pyplot as plt

    from env import AGENT_STEP, HEIGHT, WIDTH, draw_environment, in_target, is_collision

    xs = np.arange(resolucion / 2, WIDTH, resolucion)
    ys = np.arange(resolucion / 2, HEIGHT, resolucion)
    valor = np.full((len(ys), len(xs)), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if not is_collision(x, y) and not in_target(x, y):
                valor[i, j] = agente.q((x, y)).max()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(valor, origin="lower", extent=(0, WIDTH, 0, HEIGHT), cmap="viridis")
    fig.colorbar(im, ax=ax, label="max_a q̂(s, a, w)")
    draw_environment(ax)
    flechas = np.arange(0.5, WIDTH, 0.5)
    dx = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
    for x in flechas:
        for y in flechas:
            if not is_collision(x, y) and not in_target(x, y):
                u, v = dx[agente.accion_greedy((x, y))]
                ax.arrow(x, y, u * AGENT_STEP, v * AGENT_STEP, head_width=0.08)
    ax.set_title(f"{nombre}: valor y política greedy")


def visualiza(agente, n):
    env = NavigationEnv(render_mode="human")
    for posicion in posiciones_aleatorias(n, seed=1):
        obs, _ = env.reset(options={"posicion": posicion})
        acciones = []
        while True:
            accion = agente.accion_greedy(obs)
            acciones.append(ACTION_NAMES[accion])
            obs, _, terminated, truncated, _ = env.step(accion)
            env.render()
            if terminated or truncated:
                break
        final = (
            "objetivo" if env.target else "choque" if env.collision else "sin llegar"
        )
        x, y = posicion
        print(f"Desde ({x:.2f}, {y:.2f}): {env.steps} pasos, {final}")
    env.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.guarda and args.grupo == "sin_grupo":
        parser.error("con --guarda hay que indicar el grupo: --grupo G7")
    nombres = list(ALGORITMOS) if args.algoritmo == "ambos" else [args.algoritmo]

    agentes, historiales = {}, {}
    for nombre in nombres:
        agentes[nombre], historiales[nombre] = entrena_uno(nombre, args)

    print("\n=== Política greedy en las 100 posiciones de validación ===")
    for nombre, agente in agentes.items():
        informa_evaluacion(nombre, agente)

    if args.guarda:
        if args.episodios > PRESUPUESTO_EPISODIOS:
            print(
                f"AVISO: {args.episodios} episodios superan el presupuesto de "
                f"{PRESUPUESTO_EPISODIOS}: este agente no entra en la leaderboard."
            )
        Path(args.guarda).parent.mkdir(parents=True, exist_ok=True)
        for nombre, agente in agentes.items():
            ruta = f"{args.guarda}_{nombre}.npz"
            agente.guarda(
                ruta,
                grupo=args.grupo,
                algoritmo=nombre,
                episodios=args.episodios,
                alpha=args.alpha,
                gamma=args.gamma,
                q0=args.q0,
                seed=args.seed,
                version=f"Python {platform.python_version()} · numpy "
                f"{np.__version__} · {platform.system()} {platform.machine()}",
            )
            print(f"Pesos guardados en {ruta}")

    if args.render:
        for agente in agentes.values():
            visualiza(agente, args.render)

    if args.graficas or args.mapa:
        import matplotlib.pyplot as plt

        if args.graficas:
            dibuja_graficas(historiales)
        if args.mapa:
            for nombre, agente in agentes.items():
                dibuja_mapa(nombre, agente)
        plt.show()


if __name__ == "__main__":
    main()
