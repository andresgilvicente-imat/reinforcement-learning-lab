"""Resume un CSV de barrido.py: media y peor caso entre semillas por configuración.

    python experiments/analiza.py experiments/resultados/e1_sarsa.csv [--top 15] [--figura]

Se ordena por la media de llegadas entre semillas (clasificación 1) y se
muestra también el peor caso (robustez) y los pasos de más (clasificación 2).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CLAVES = ("algoritmo", "rho", "n_tiles", "num_tilings", "q0", "eps", "recompensa")

DEFECTOS = {"recompensa": "base"}


def _v(r, k):
    """Valor de una columna; los CSV antiguos no tienen 'recompensa'."""
    return r.get(k) or DEFECTOS.get(k, "")


def tiles_txt(n):
    """'12' -> '12×12'; '12x16' -> '12×16'."""
    p = str(n).split("x")
    return f"{p[0]}×{p[-1]}"


def tiles_args(n):
    p = str(n).split("x")
    return f"{p[0]} {p[-1]}"


def _tk(n):
    return tuple(int(v) for v in str(n).split("x"))


def carga(rutas):
    filas = []
    for ruta in rutas:
        with open(ruta, newline="") as f:
            filas += list(csv.DictReader(f))
    return filas


def agrupa(filas):
    grupos = defaultdict(list)
    for r in filas:
        grupos[tuple(_v(r, k) for k in CLAVES)].append(r)
    resumen = []
    for clave, rs in grupos.items():
        lleg = np.array([float(r["llegadas"]) for r in rs])
        extra = np.array([float(r["extra"]) if r["extra"] not in ("", "nan") else np.nan
                          for r in rs])
        resumen.append(dict(
            zip(CLAVES, clave), n=len(rs),
            diverge=sum(int(r["diverge"]) for r in rs),
            lleg_media=lleg.mean(), lleg_min=lleg.min(), lleg_std=lleg.std(),
            extra_media=np.nanmean(extra) if np.isfinite(extra).any() else np.nan,
        ))
    return resumen


def tabla(resumen, top, orden):
    if orden == "extra":
        # Clasificación 2: solo cuentan agentes que llegan desde >= 50 %
        resumen = [r for r in resumen if r["lleg_min"] >= 500]
        resumen.sort(key=lambda r: r["extra_media"])
    else:
        resumen.sort(key=lambda r: (-r["lleg_media"], r["extra_media"]))
    print(f"{'rho':>5} {'alpha':>8} {'tiles':>5} {'recomp.':>8} {'tilings':>7} {'q0':>5} "
          f"{'eps':>26} {'div':>3} {'lleg media':>10} {'min':>5} {'std':>5} {'extra':>6}")
    for r in resumen[:top]:
        alpha = float(r["rho"]) / int(r["num_tilings"])
        print(f"{r['rho']:>5} {alpha:8.4f} {r['n_tiles']:>5} {r['recompensa']:>8} {r['num_tilings']:>7} "
              f"{r['q0']:>5} {r['eps']:>26} {r['diverge']:>3} {r['lleg_media']:10.1f} "
              f"{r['lleg_min']:5.0f} {r['lleg_std']:5.1f} {r['extra_media']:6.2f}")


def figura(resumen, salida):
    """Mapa de calor llegadas medias: tiles x tilings, un panel por rho."""
    import matplotlib.pyplot as plt

    rhos = sorted({float(r["rho"]) for r in resumen})
    tiles = sorted({r["n_tiles"] for r in resumen}, key=_tk)
    tilings = sorted({int(r["num_tilings"]) for r in resumen})
    fig, axes = plt.subplots(1, len(rhos), figsize=(3.6 * len(rhos), 3.4),
                             squeeze=False, constrained_layout=True)
    for ax, rho in zip(axes[0], rhos):
        m = np.full((len(tilings), len(tiles)), np.nan)
        for r in resumen:
            if float(r["rho"]) == rho:
                m[tilings.index(int(r["num_tilings"])), tiles.index(r["n_tiles"])] = \
                    r["lleg_media"] / 10  # % sobre 1000
        im = ax.imshow(m, vmin=0, vmax=100, cmap="viridis", origin="lower")
        for i in range(len(tilings)):
            for j in range(len(tiles)):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center",
                            color="white" if m[i, j] < 60 else "black", fontsize=8)
        ax.set_xticks(range(len(tiles)), [tiles_txt(t) for t in tiles])
        ax.set_yticks(range(len(tilings)), tilings)
        ax.set_xlabel("tiles por dimensión")
        ax.set_ylabel("tilings")
        ax.set_title(f"ρ = α·tilings = {rho}")
    fig.colorbar(im, ax=axes[0].tolist(), label="% llegadas (media de semillas)")
    fig.suptitle(f"{resumen[0]['algoritmo']}: llegadas en posiciones de selección")
    fig.savefig(salida, dpi=130)
    print(f"Figura: {salida}")


def _nombre_eps(eps):
    if eps == "codigo":
        return "código\n(al ejecutar)"
    ini, fin, f1, f2 = (float(v) for v in eps.strip("()").split(","))
    return f"{ini:g}→{fin:g}\n[{f1:g}, {f2:g}]"


def figura_eps(resumen, salida):
    """Etapas 2-3: diseño x programa de epsilon, llegadas y pasos de más."""
    import matplotlib.pyplot as plt

    disenos = sorted({(float(r["rho"]), r["n_tiles"], int(r["num_tilings"]), r["recompensa"])
                      for r in resumen}, reverse=True)
    epss = sorted({r["eps"] for r in resumen},
                  key=lambda e: (e != "codigo", [-float(v) for v in
                                                 e.strip("()").split(",")] if e != "codigo" else []))
    lleg = np.full((len(disenos), len(epss)), np.nan)
    extra = np.full_like(lleg, np.nan)
    div = np.zeros_like(lleg)
    for r in resumen:
        i = disenos.index((float(r["rho"]), r["n_tiles"], int(r["num_tilings"]), r["recompensa"]))
        j = epss.index(r["eps"])
        lleg[i, j], extra[i, j], div[i, j] = r["lleg_media"] / 10, r["extra_media"], r["diverge"]

    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(epss) + 5, 0.55 * len(disenos) + 2.6),
                             constrained_layout=True)
    paneles = [(axes[0], lleg, "% llegadas (media de semillas) ↑", "viridis", 90, 100, "{:.1f}"),
               (axes[1], extra, "pasos de más (media) ↓", "viridis_r", 0, 1.0, "{:.2f}")]
    for ax, m, titulo, cmap, vmin, vmax, fmt in paneles:
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    txt = fmt.format(m[i, j]) + (f"\n({div[i, j]:.0f} div)" if div[i, j] else "")
                    oscuro = (m[i, j] - vmin) / (vmax - vmin)
                    oscuro = oscuro < 0.5 if cmap == "viridis" else oscuro > 0.5
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                            color="white" if oscuro else "black")
        ax.set_xticks(range(len(epss)), [_nombre_eps(e) for e in epss], fontsize=7.5)
        ax.set_yticks(range(len(disenos)),
                      [f"ρ={d[0]:g} · {tiles_txt(d[1])} · {d[2]} til." + (" · shaping" if d[3] == "shaping" else "") for d in disenos],
                      fontsize=8)
        ax.set_xlabel("programa de ε: inicial→final [fin meseta, fin decaimiento]")
        ax.set_title(titulo)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{resumen[0]['algoritmo']}: diseño × programa de ε "
                 "(celdas vacías = combinación no probada)")
    fig.savefig(salida, dpi=130)
    print(f"Figura: {salida}")


def figura_programas(resumen, salida, n=2000):
    """Dibuja la forma de cada programa de epsilon probado."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    k = np.arange(n)
    for eps in sorted({r["eps"] for r in resumen}):
        if eps == "codigo":
            continue
        ini, fin, f1, f2 = (float(v) for v in eps.strip("()").split(","))
        a, b = int(f1 * n), int(f2 * n)
        prog = np.clip((k - a) / max(1, b - a - 1), 0, 1)
        y = ini * (1 - prog) if fin <= 0 else ini * (fin / ini) ** prog
        ax.plot(k, y, label=_nombre_eps(eps).replace("\n", " "))
    ax.set_xlabel("episodio")
    ax.set_ylabel("ε")
    ax.set_title("Programas de ε probados")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(salida, dpi=130)
    print(f"Figura: {salida}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--orden", choices=["llegadas", "extra"], default="llegadas")
    p.add_argument("--figura", action="store_true")
    a = p.parse_args()
    resumen = agrupa(carga(a.csv))
    tabla(resumen, a.top, a.orden)
    if a.figura:
        base = Path(a.csv[0]).with_suffix("")
        if len({r["eps"] for r in resumen}) > 1:  # etapas 2-3: varía el programa de ε
            nombre = base.name + "".join("+" + Path(c).stem.split("_")[0] for c in a.csv[1:])
            figura_eps(resumen, base.with_name(nombre + "_eps.png"))
            figura_programas(resumen, base.with_name(nombre + "_programas.png"))
        else:  # etapa 1: varía el diseño del tile coding
            figura(resumen, base.with_suffix(".png"))


if __name__ == "__main__":
    main()
