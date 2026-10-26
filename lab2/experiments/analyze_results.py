"""Genera las figuras y tablas del informe a partir de results/raw.

Uso (desde lab2/experiments, después de run_experiments.py):
    python analyze_results.py

Salidas:
  figures/<categoria>/*.pdf|png   todas las figuras (las del informe se copian a report/figures)
  report/tables/*.tex             tablas LaTeX que incluye report.tex
  results/summaries/*.csv|json    cifras que se citan en el texto
"""

from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import (  # noqa: E402
    DR,
    FIGURES,
    GAMMA,
    INITIAL_STATES,
    LAB2,
    NA,
    RESULTS,
    UR,
    bootstrap_ci,
    expected_training_return,
    goal_reachable,
    make_env,
    optimal_action_sets,
    q_star,
    reachable_states,
    sign_flip_pvalue,
    soft_optimal_q,
    stationary_sd_sarsa,
)
from loaders import first_stable, load_runs, stack  # noqa: E402

# ------------------------------------------------------------------ estilo
C_SARSA, C_QL = "#2a78d6", "#eb6834"
C_UR, C_DR = "#1baf7a", "#6250d6"
C_REF = "#52514e"
ALGO_C = {"sarsa": C_SARSA, "q_learning": C_QL}
ALGO_L = {"sarsa": "SARSA", "q_learning": "Q-learning"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
})
W = 6.3  # ancho de texto (pulgadas)

REPORT = LAB2 / "report"
TABLES = REPORT / "tables"
SUMM = RESULTS / "summaries"
for d in (TABLES, SUMM, REPORT / "figures"):
    d.mkdir(parents=True, exist_ok=True)

P = make_env().unwrapped.P
QS = q_star(P)
QE = soft_optimal_q(P, GAMMA, 0.2)[0]
REACH = reachable_states(P)
GOAL_OK = goal_reachable(P)
ASTAR = optimal_action_sets(QS)
NUMBERS = {}  # cifras citadas en el texto


def save(fig, cat, name, report=True):
    out = FIGURES / cat
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    if report:
        shutil.copy(out / f"{name}.pdf", REPORT / "figures" / f"{name}.pdf")
    plt.close(fig)


def on_grid(run, values, grid):
    """Valor de una serie por episodio evaluado en pasos globales ``grid``."""
    idx = np.searchsorted(run["cum_steps"], grid, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "$-$")


# ================================================================= figura 1
def fig_map():
    from env import CELL_TO_STATE, GRID_WORLD

    nrow, ncol = len(GRID_WORLD), len(GRID_WORLD[0])
    fig, ax = plt.subplots(figsize=(2.7, 3.5))
    for r in range(nrow):
        for c in range(ncol):
            cell, ch = r * ncol + c, GRID_WORLD[r][c]
            y = nrow - 1 - r
            if ch == "X":
                fc, lab, tc = "#0b0b0b", "", "w"
            elif ch == "G":
                fc, lab, tc = "#008300", "G", "w"
            else:
                s = CELL_TO_STATE[cell]
                if s not in REACH:
                    fc = "#d9d8d4"
                elif s not in GOAL_OK:
                    fc = "#f4c7c3"
                else:
                    fc = "#cfe2f7"
                lab, tc = f"$s_{{{s}}}$", "#0b0b0b"
            ax.add_patch(Rectangle((c, y), 1, 1, fc=fc, ec="white", lw=1.5))
            if lab:
                ax.text(c + 0.5, y + 0.62, lab, ha="center", va="center", fontsize=8, color=tc)
            if ch == "F":
                s = CELL_TO_STATE[cell]
                acts = ASTAR[s]
                arrows = "".join("↗" if a == UR else "↘" for a in sorted(acts))
                if len(acts) == 2:
                    same = P[s][0][0][1:3] == P[s][1][0][1:3]
                    arrows = "=" if same else "↗↘"
                ax.text(c + 0.5, y + 0.27, arrows, ha="center", va="center", fontsize=8,
                        color="#0b0b0b", fontfamily="DejaVu Sans")
                if s in INITIAL_STATES:
                    ax.add_patch(Rectangle((c + 0.06, y + 0.06), 0.88, 0.88, fc="none",
                                           ec="#0b0b0b", lw=1.1))
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in ("#cfe2f7", "#f4c7c3", "#d9d8d4")]
    ax.legend(handles, ["alcanzable, con ruta al objetivo", "alcanzable, sin ruta al objetivo",
                        "inalcanzable desde $s_{init}$"],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=6.5, ncol=1)
    save(fig, "baseline", "fig_map")


# ================================================================= tabla 1
def table_qstar():
    rows = []
    for s in range(20):
        acts = ASTAR[s]
        same = P[s][0][0][1:3] == P[s][1][0][1:3]
        notes = []
        if s not in REACH:
            notes.append("inalcanzable")
        if s not in GOAL_OK:
            notes.append("sin ruta al objetivo")
        if len(acts) == 2:
            notes.append("misma transición" if same else "empate, rutas distintas")
        a_txt = r"$\nearrow,\searrow$" if len(acts) == 2 else (r"$\nearrow$" if UR in acts else r"$\searrow$")
        rows.append(f"$s_{{{s}}}$ & {fmt(QS[s, UR], 4)} & {fmt(QS[s, DR], 4)} & {a_txt} & {', '.join(notes)} \\\\")
    body = "\n".join(rows)
    (TABLES / "tab_qstar.tex").write_text(
        "\\begin{tabular}{crrcl}\n\\toprule\n"
        "Estado & $q_*(s,\\nearrow)$ & $q_*(s,\\searrow)$ & $\\mathcal{A}_*(s)$ & Observación \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n", encoding="utf-8")
    pd.DataFrame(QS[:20], columns=["q*(s,UR)", "q*(s,DR)"]).rename_axis("s").to_csv(SUMM / "q_star.csv")
    pd.DataFrame(QE[:20], columns=["q_eps(s,UR)", "q_eps(s,DR)"]).rename_axis("s").to_csv(SUMM / "q_eps_0.2.csv")


# ================================================================= figura 2
def fig_q_evolution():
    grid = np.unique(np.round(np.logspace(0.5, np.log10(80_000), 500)).astype(int))
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.4), sharex=True)
    for j, algo in enumerate(("sarsa", "q_learning")):
        R = load_runs("base", "base", algo)
        track = list(R[0]["track"])
        for i, s in enumerate((6, 13)):
            ax = axes[i, j]
            k = track.index(s)
            for a, col, sym in ((UR, C_UR, r"\nearrow"), (DR, C_DR, r"\searrow")):
                band = np.array([on_grid(r, r["q_track"][:, k, a].astype(float), grid) for r in R])
                band = np.where(np.isnan(band), 0.0, band)
                lo, hi = np.percentile(band, [10, 90], axis=0)
                ax.fill_between(grid, lo, hi, color=col, alpha=0.18, lw=0)
                ax.plot(grid, on_grid(R[0], R[0]["q_track"][:, k, a].astype(float), grid), color=col,
                        lw=1.0)
                ax.axhline(QS[s, a], color=col, ls=":", lw=1.0)
                if algo == "sarsa" and not np.isclose(QE[s, a], QS[s, a]):
                    ax.axhline(QE[s, a], color=col, ls="--", lw=1.0)
            ax.set_xscale("log")
            ax.set_title(f"{ALGO_L[algo]} · $s_{{{s}}}$")
            if j == 0:
                ax.set_ylabel(r"$Q(s,a)$")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=C_UR, lw=1.2), Line2D([], [], color=C_DR, lw=1.2),
               Patch(color=C_REF, alpha=0.25), Line2D([], [], color=C_REF, ls=":"),
               Line2D([], [], color=C_REF, ls="--")]
    labels = [r"$a=\nearrow$ (semilla 0)", r"$a=\searrow$ (semilla 0)", "percentiles 10–90 (20 semillas)",
              r"$q_*(s,a)$", r"$q_\varepsilon(s,a)$ ($\varepsilon=0.2$)"]
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.8, bbox_to_anchor=(0.5, 1.03))
    fig.supxlabel("pasos globales de entrenamiento (escala logarítmica)", fontsize=9, y=0.01)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "states", "fig_q_s6_s13")

    # cifras de fases para el texto
    out = {}
    for algo in ("sarsa", "q_learning"):
        R = load_runs("base", "base", algo)
        tm = stack(R, "tail_mean")
        ts = stack(R, "tail_sd")
        first_star = np.array([r["cum_steps"][r["first_star"][[6, 13], :]] for r in R])
        first_eps = np.array([r["cum_steps"][r["first_eps"][[6, 13], :]] for r in R])
        vis_first = np.array([[r["visits_sub"][min(r["first_star"][s, a] // 10, 1999), s, a]
                               for s, a in ((6, DR), (13, UR))] for r in R])
        minq6 = np.array([r["q_track"][:, list(r["track"]).index(6), UR].min() for r in R])
        out[algo] = {
            "tail_mean_s6": tm[:, 6].mean(0).tolist(), "tail_mean_s13": tm[:, 13].mean(0).tolist(),
            "tail_sd_s13": ts[:, 13].mean(0).tolist(),
            "median_steps_first_star_s6_s13": np.median(first_star, 0).tolist(),
            "median_steps_first_eps_s6_s13": np.median(first_eps, 0).tolist(),
            "min_Q_s6_UR_mean": float(minq6.mean()),
            "approx_visits_at_first_star_s6DR_s13UR": np.median(vis_first, 0).tolist(),
            "median_visits_s6_s13": np.median(stack(R, "visits")[:, [6, 13]], 0).tolist(),
        }
    NUMBERS["s6_s13"] = out


# ================================================================= figura 3 + tabla 3
def fig_alpha_s13():
    alphas = [0.05, 0.1, 0.5]
    src = {0.05: ("alpha", "alpha_0.05"), 0.1: ("base", "base"), 0.5: ("alpha", "alpha_0.5"),
           0.01: ("alpha", "alpha_0.01"), 1.0: ("alpha", "alpha_1.0")}
    k13 = 4  # índice de s13 en TRACK
    fig = plt.figure(figsize=(W, 4.3))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.2, 1], hspace=0.35, wspace=0.08)
    ep = np.arange(1, 20_001)
    rng_bins = np.linspace(-6.5, 3.0, 60)
    for i, al in enumerate(alphas):
        R = load_runs(*src[al], "sarsa")[:10]
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(ep, R[0]["q_track"][:, k13, UR], color=C_SARSA, lw=0.6, label="SARSA (semilla 0)")
        Rq = load_runs(*src[al], "q_learning")[:10]
        ax.plot(ep, Rq[0]["q_track"][:, k13, UR], color=C_QL, lw=1.0, label="Q-learning (semilla 0)")
        ax.axhline(QS[13, UR], color=C_REF, ls=":", lw=1)
        ax.axhline(QE[13, UR], color=C_REF, ls="--", lw=1)
        ax.set_ylim(-6.6, 3.2)
        ax.set_ylabel(rf"$\alpha={al}$")
        if i == 0:
            ax.set_title(r"$Q(s_{13},\nearrow)$ por episodio ($\varepsilon=0.2$, $Q_0=0$)")
            ax.legend(loc="lower right", ncol=2, fontsize=6.5)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("episodio")
        axh = fig.add_subplot(gs[i, 1], sharey=ax)
        tail = np.concatenate([r["q_track"][15_000:, k13, UR] for r in R])
        axh.hist(tail, bins=rng_bins, orientation="horizontal", density=True, color=C_SARSA, alpha=0.6)
        sd, _ = stationary_sd_sarsa(P, QE, 13, UR, al, 0.2)
        yy = np.linspace(-6.5, 3, 300)
        if al <= 0.1:
            axh.plot(np.exp(-0.5 * ((yy - QE[13, UR]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), yy,
                 color="#0b0b0b", lw=0.9)
        axh.tick_params(labelleft=False)
        if i == 0:
            axh.set_title("últimos 5000 ep.", fontsize=8)
        if i == 2:
            axh.set_xlabel("densidad")
    for ax in fig.axes:
        ax.grid(alpha=0.2)
    save(fig, "hyperparameters", "fig_alpha_s13")

    rows, data = [], []
    for al in (0.01, 0.05, 0.1, 0.5, 1.0):
        R = load_runs(*src[al], "sarsa")[:10]
        Rq = load_runs(*src[al], "q_learning")[:10]
        tail = np.concatenate([r["q_track"][15_000:, k13, UR] for r in R])
        sd_th, _ = stationary_sd_sarsa(P, QE, 13, UR, al, 0.2)
        ev_s = stack(R, "eval_ret")[:, -1].mean()
        ev_q = stack(Rq, "eval_ret")[:, -1].mean()
        pol_s = [first_stable(o == len(REACH)) for o in stack(R, "ok_star")]
        pol_q = [first_stable(o == len(REACH)) for o in stack(Rq, "ok_star")]
        qq = stack(Rq, "tail_mean")[:, 13, UR].mean()
        drop = al * (QE[13, UR] - (-1 + GAMMA * QE[11, DR]))
        data.append(dict(alpha=al, sd_theory=sd_th, sd_sarsa=tail.std(), mean_sarsa=tail.mean(),
                         drop_theory=drop, q_ql=qq, eval_sarsa=ev_s, eval_ql=ev_q,
                         pol_sarsa_median=np.median([p if p >= 0 else np.inf for p in pol_s]),
                         pol_ql_median=np.median([p if p >= 0 else np.inf for p in pol_q])))
        pol_txt = lambda ps: (f"{int(np.median(ps)):,}".replace(",", "\\,") if all(p >= 0 for p in ps)  # noqa: E731
                              else f"{sum(p < 0 for p in ps)}/10 nunca")
        rows.append(f"{al} & {fmt(sd_th)} & {fmt(tail.std())} & {fmt(tail.mean())} & {fmt(qq)} & "
                    f"{fmt(ev_s)} & {fmt(ev_q)} & {pol_txt(pol_s)} & {pol_txt(pol_q)} \\\\")
    pd.DataFrame(data).to_csv(SUMM / "alpha_s13.csv", index=False)
    (TABLES / "tab_alpha.tex").write_text(
        "\\begin{tabular}{lcccccccc}\n\\toprule\n"
        " & \\multicolumn{3}{c}{SARSA: $Q(s_{13},\\nearrow)$} & QL & \\multicolumn{2}{c}{Retorno greedy} "
        "& \\multicolumn{2}{c}{Política estable (ep.)} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
        "$\\alpha$ & $\\sigma$ teórica & $\\sigma$ emp. & media emp. & media & SARSA & QL & SARSA & QL \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


# ================================================================= figura 4 + tabla 4
def fig_returns():
    Rs, Rq = load_runs("base", "base", "sarsa"), load_runs("base", "base", "q_learning")
    pi = QS.argmax(1).copy()
    pi[10] = UR
    e_up = expected_training_return(P, pi, 0.2)
    pi[10] = DR
    e_dn = expected_training_return(P, pi, 0.2)
    NUMBERS["expected_train_return_eps0.2"] = {"s10_UR": e_up, "s10_DR": e_dn}

    fig, axes = plt.subplots(3, 1, figsize=(W, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 0.8]})
    w = 100
    for R, algo in ((Rs, "sarsa"), (Rq, "q_learning")):
        ret = stack(R, "returns").astype(float)
        cs = np.cumsum(ret, axis=1)
        ma = np.full_like(ret, np.nan)
        ma[:, w - 1:] = (cs[:, w - 1:] - np.concatenate([np.zeros((len(R), 1)), cs[:, :-w]], 1)) / w
        m, se = ma.mean(0), ma.std(0) / np.sqrt(len(R))
        # puntos equiespaciados en escala log (el eje x es logarítmico)
        idx = np.unique(np.round(np.logspace(0, np.log10(ret.shape[1]), 1500)).astype(int)) - 1
        x = idx + 1
        axes[0].plot(x, m[idx], color=ALGO_C[algo], lw=1, label=ALGO_L[algo])
        axes[0].fill_between(x, (m - 1.96 * se)[idx], (m + 1.96 * se)[idx], color=ALGO_C[algo],
                             alpha=0.2, lw=0)
        er = stack(R, "eval_ret").mean(2)
        axes[1].plot(R[0]["eval_ep"], er.mean(0), color=ALGO_C[algo], lw=1, label=ALGO_L[algo])
        g10 = stack(R, "greedy_s10")
        axes[2].plot(x, (g10 == DR).mean(0)[idx], color=ALGO_C[algo], lw=1, label=ALGO_L[algo])
    axes[0].axhline(e_up, color=C_REF, ls="--", lw=0.9)
    axes[0].axhline(e_dn, color=C_REF, ls=":", lw=0.9)
    axes[0].text(1.3, e_up + 0.05, r"teórico, $\pi_*$ con $s_{10}\nearrow$", fontsize=6.5, color=C_REF)
    axes[0].text(1.3, e_dn - 0.17, r"teórico, $\pi_*$ con $s_{10}\searrow$", fontsize=6.5, color=C_REF)
    axes[0].set_ylabel("retorno de entrenamiento\n(media móvil 100 ep.)")
    axes[0].set_ylim(-3.2, 0.0)
    axes[0].legend(loc="lower right")
    axes[1].axhline(1.5, color=C_REF, ls=":", lw=0.9)
    axes[1].set_ylabel("retorno greedy\n($\\varepsilon=0$, media $s_{init}$)")
    axes[2].set_ylabel("fracción de semillas\ncon greedy en $s_{10}$\n= abajo-derecha")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("episodio (escala log.)")
    axes[2].set_xscale("log")
    fig.align_ylabels(axes)
    fig.tight_layout()
    save(fig, "comparisons", "fig_returns")

    phases = [(0, 100), (100, 500), (500, 10_000), (10_000, 20_000)]
    rows, data = [], []
    for a, b in phases:
        s_ = stack(Rs, "returns")[:, a:b].mean(1)
        q_ = stack(Rq, "returns")[:, a:b].mean(1)
        d = s_ - q_
        m, lo, hi = bootstrap_ci(d)
        p = sign_flip_pvalue(d)
        frac = (stack(Rq, "greedy_s10")[:, a:b] == DR).mean()
        fracs = (stack(Rs, "greedy_s10")[:, a:b] == DR).mean()
        data.append(dict(phase=f"[{a},{b})", sarsa=s_.mean(), ql=q_.mean(), diff=m, lo=lo, hi=hi, p=p,
                         ql_s10_dr=frac, sarsa_s10_dr=fracs, sarsa_wins=int((d > 0).sum())))
        p_txt = "$<0.001$" if p < 0.001 else f"{p:.3f}"
        # el {} evita que "\\ [" se lea como \\[<longitud>]
        rng_txt = "{}" + f"[{a:_}, {b:_})".replace("_", "\\,")
        rows.append(f"{rng_txt} & {fmt(s_.mean())} & {fmt(q_.mean())} & "
                    f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}] & {p_txt} & {int((d > 0).sum())}/20 & "
                    f"{100 * fracs:.0f}\\,\\% & {100 * frac:.0f}\\,\\% \\\\")
    pd.DataFrame(data).to_csv(SUMM / "returns_by_phase.csv", index=False)
    (TABLES / "tab_returns.tex").write_text(
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Retorno medio} & & & & \\multicolumn{2}{c}{greedy$(s_{10})=\\searrow$} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){7-8}\n"
        "Episodios & SARSA & QL & Dif. [IC 95\\,\\%] & $p$ & SARSA mejor & SARSA & QL \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


# ================================================================= figura 5
def fig_epsilon():
    eps_cfg = {0.0: ("epsilon", "eps_0.0"), 0.05: ("epsilon", "eps_0.05"), 0.1: ("epsilon", "eps_0.1"),
               0.2: ("base", "base"), 0.5: ("epsilon", "eps_0.5")}
    eg = np.linspace(0, 0.6, 61)
    pi = QS.argmax(1).copy()
    pi[10] = UR
    th_ret = [expected_training_return(P, pi, e) for e in eg]
    th_q13 = [soft_optimal_q(P, GAMMA, e)[0][13, UR] for e in eg]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3))
    data = []
    for algo, off in (("sarsa", -0.006), ("q_learning", 0.006)):
        xs, r_m, r_sd, ev, q13, err, err_ng = [], [], [], [], [], [], []
        for e, src in eps_cfg.items():
            R = load_runs(*src, algo)[:10]
            ret = stack(R, "returns")[:, 10_000:].mean(1)
            xs.append(e)
            r_m.append(ret.mean())
            r_sd.append(ret.std())
            ev.append(stack(R, "eval_ret")[:, -1].mean())
            q13.append(stack(R, "tail_mean")[:, 13, UR].mean())
            qf = stack(R, "q_final")
            e_all = np.abs(qf - QS)[:, REACH]
            err.append(np.median(e_all.max(axis=(1, 2))))
            cover = (stack(R, "visits")[:, REACH] > 0).sum(axis=(1, 2)).mean()
            data.append(dict(algo=algo, epsilon=e, train_return_last10k=ret.mean(), sd=ret.std(),
                             eval_return=ev[-1], q13_ur_tail=q13[-1], max_err_qstar=err[-1],
                             coverage_pairs=cover, success_last10k=stack(R, "success")[:, 10_000:].mean(),
                             q6_ur_final=qf[:, 6, UR].mean()))
        xs = np.array(xs)
        axes[0].errorbar(xs + off, r_m, yerr=r_sd, color=ALGO_C[algo], marker="o", ms=3, lw=1,
                         capsize=2, label=f"{ALGO_L[algo]}: entrenamiento")
        axes[0].plot(xs + off, ev, color=ALGO_C[algo], marker="s", ms=3, lw=0.8, ls="--",
                     label=f"{ALGO_L[algo]}: greedy")
        axes[1].plot(xs + off, q13, color=ALGO_C[algo], marker="o", ms=3, lw=1, label=ALGO_L[algo])
        axes[2].plot(xs + off, err, color=ALGO_C[algo], marker="o", ms=3, lw=1, label=ALGO_L[algo])
    axes[0].plot(eg, th_ret, color=C_REF, lw=0.8, ls=":")
    from matplotlib.lines import Line2D
    axes[0].legend([Line2D([], [], color=C_SARSA, lw=1.2), Line2D([], [], color=C_QL, lw=1.2),
                    Line2D([], [], color=C_REF, marker="o", ms=3, lw=1),
                    Line2D([], [], color=C_REF, marker="s", ms=3, ls="--", lw=0.8),
                    Line2D([], [], color=C_REF, ls=":", lw=0.8)],
                   ["SARSA", "Q-learning", "entrenamiento", r"greedy ($\varepsilon=0$)", r"teórico ($\pi_*$, $s_{10}\nearrow$)"],
                   fontsize=5.8, loc="lower left")
    axes[0].set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel("retorno medio (ep. 10k–20k)")
    axes[1].plot(eg, th_q13, color=C_REF, lw=0.8, ls="--", label=r"$q_\varepsilon(s_{13},\nearrow)$")
    axes[1].axhline(QS[13, UR], color=C_REF, lw=0.8, ls=":")
    axes[1].set_xlabel(r"$\varepsilon$")
    axes[1].set_ylabel(r"$Q(s_{13},\nearrow)$ media final")
    axes[1].legend(fontsize=6)
    axes[2].set_xlabel(r"$\varepsilon$")
    axes[2].set_ylabel(r"máx. $|Q-q_*|$ en 34 pares")
    axes[2].legend(fontsize=6)
    for ax, t in zip(axes, "abc"):
        ax.set_title(f"({t})", loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, "hyperparameters", "fig_epsilon")
    pd.DataFrame(data).to_csv(SUMM / "epsilon_sweep.csv", index=False)


# ================================================================= figura 6
def fig_qinit():
    cfgs = [("q_init", "q0_-5.0", r"$Q_0=-5$", "#e34948"), ("base", "base", r"$Q_0=0$", "#52514e"),
            ("q_init", "q0_5.0", r"$Q_0=+5$", "#008300")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True)
    data = []
    for algo, ls in (("sarsa", "-"), ("q_learning", "--")):
        for g, n, lab, col in cfgs:
            R = load_runs(g, n, algo)[:10]
            er = stack(R, "eval_ret").mean(2)
            axes[0].plot(np.maximum(R[0]["eval_ep"], 1), er.mean(0), color=col, ls=ls, lw=1,
                         label=f"{ALGO_L[algo]}, {lab}")
    for g, n, lab, col, ls in (("epsilon", "eps_0.0", r"$\varepsilon=0$, $Q_0=0$", "#52514e", "-"),
                               ("inter", "e0_q0+5", r"$\varepsilon=0$, $Q_0=+5$", "#008300", "-"),
                               ("inter", "e0_q0-5", r"$\varepsilon=0$, $Q_0=-5$", "#e34948", "-"),
                               ("inter", "e0.05_q0-5", r"$\varepsilon=0.05$, $Q_0=-5$", "#e34948", "--"),
                               ("q_init", "q0_-5.0", r"$\varepsilon=0.2$, $Q_0=-5$", "#e34948", ":")):
        R = load_runs(g, n, "q_learning")[:10]
        er = stack(R, "eval_ret").mean(2)
        axes[1].plot(np.maximum(R[0]["eval_ep"], 1), er.mean(0), color=col, ls=ls, lw=1, label=lab)
        cover = (stack(R, "visits")[:, REACH] > 0).sum(axis=(1, 2)).mean()
        data.append(dict(config=n, eval_final=er[:, -1].mean(), coverage=cover,
                         success_last10k=stack(R, "success")[:, 10_000:].mean(),
                         first_eval_opt=np.median([R[0]["eval_ep"][np.argmax(x >= 1.5 - 1e-9)]
                                                   if (x >= 1.5 - 1e-9).any() else -1 for x in er])))
    for g, n, _, _ in cfgs:
        for algo in ("sarsa", "q_learning"):
            R = load_runs(g, n, algo)[:10]
            er = stack(R, "eval_ret").mean(2)
            data.append(dict(config=f"{n}/{algo}", eval_final=er[:, -1].mean(),
                             first_eval_opt=np.median([R[0]["eval_ep"][np.argmax(x >= 1.5 - 1e-9)] for x in er]),
                             ret_first500=stack(R, "returns")[:, :500].mean(),
                             err_final=np.median(np.abs(stack(R, "q_final") - QS)[:, REACH].max(axis=(1, 2)))))
    for ax in axes:
        ax.set_xscale("log")
        ax.axhline(1.5, color=C_REF, ls=":", lw=0.8)
        ax.set_xlabel("episodio (escala log.)")
    axes[0].set_ylabel("retorno greedy medio")
    axes[0].set_title(r"(a) $\varepsilon=0.2$: efecto de $Q_0$", loc="left", fontsize=8)
    axes[1].set_title(r"(b) Q-learning con poca exploración", loc="left", fontsize=8)
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.tight_layout()
    save(fig, "hyperparameters", "fig_qinit")
    pd.DataFrame(data).to_csv(SUMM / "qinit_exploration.csv", index=False)


# ================================================================= figura 7
def fig_visits_error():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    cfgs = [("base", "base"), ("epsilon", "eps_0.05"), ("epsilon", "eps_0.1"), ("epsilon", "eps_0.5")]
    n = np.logspace(0, 4, 100)
    for ax, algo, ref, lab in ((axes[0], "q_learning", QS, r"$|Q-q_*|$"),
                               (axes[1], "sarsa", None, r"$|Q-q_\varepsilon|$")):
        xs, ys = [], []
        for g, nm in cfgs:
            R = load_runs(g, nm, algo)[:10]
            eps = R[0]["params_eps"] if "params_eps" in R[0] else None  # noqa: F841
            for r in R:
                rf = QS if ref is not None else r["qeps"]
                for s in REACH:
                    for a in range(NA):
                        v = r["visits"][s, a]
                        if v == 0 or np.isclose(rf[s, a], 0):
                            continue
                        xs.append(v)
                        ys.append(max(abs(r["q_final"][s, a] - rf[s, a]) / abs(rf[s, a]), 1e-17))
        ax.scatter(xs, ys, s=4, color=ALGO_C[algo], alpha=0.35, lw=0, label=ALGO_L[algo])
        ax.plot(n, 0.9 ** n, color="#0b0b0b", lw=0.9, label=r"$(1-\alpha)^{N}$, $\alpha=0.1$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 3)
        ax.set_xlabel(r"visitas $N(s,a)$")
        ax.set_title(f"({'ab'[algo == 'sarsa']}) {ALGO_L[algo]}: error relativo {lab}/|ref|", loc="left",
                     fontsize=8)
        ax.legend(fontsize=6.5, loc="lower left")
    axes[0].set_ylabel("error relativo final")
    fig.tight_layout()
    save(fig, "comparisons", "fig_visits_error")


# ================================================================= figura 8
def fig_efficiency():
    src = {0.01: ("alpha", "alpha_0.01"), 0.05: ("alpha", "alpha_0.05"), 0.1: ("base", "base"),
           0.5: ("alpha", "alpha_0.5"), 1.0: ("alpha", "alpha_1.0")}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), sharey=True)
    data = []
    for algo in ("sarsa", "q_learning"):
        m1, m2, m3 = [], [], []
        for al, s in src.items():
            R = load_runs(*s, algo)[:10]
            steps_eval, steps_pol, steps_q = [], [], []
            for r in R:
                er = r["eval_ret"].mean(1)
                k = first_stable(er >= 1.5 - 1e-9)
                steps_eval.append(r["cum_steps"][max(r["eval_ep"][k] - 1, 0)] if k >= 0 else np.inf)
                kp = first_stable(r["ok_star"] == len(REACH))
                steps_pol.append(r["cum_steps"][kp] if kp >= 0 else np.inf)
                ref = "first_star" if algo == "q_learning" else "first_eps"
                f = r[ref][13, UR]
                steps_q.append(r["cum_steps"][f] if f >= 0 else np.inf)
            m1.append(np.median(steps_eval))
            m2.append(np.median(steps_pol))
            m3.append(np.median(steps_q))
            data.append(dict(algo=algo, alpha=al, steps_eval_stable=m1[-1], steps_policy_stable=m2[-1],
                             steps_q13_band=m3[-1], n_never_policy=int(np.sum(np.isinf(steps_pol)))))
        al = list(src)
        for ax, m in zip(axes, (m1, m2, m3)):
            mm = np.array(m, float)
            fin = np.isfinite(mm)
            ax.plot(np.array(al)[fin], mm[fin], marker="o", ms=3.5, color=ALGO_C[algo], label=ALGO_L[algo])
            for x in np.array(al)[~fin]:
                ax.plot([x], [2e5], marker="x", ms=6, color=ALGO_C[algo])
    titles = ["(a) greedy óptimo desde $s_{init}$", "(b) greedy óptimo en 17 estados",
              r"(c) $Q(s_{13},\nearrow)$ en banda $\pm0.1$"]
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(t, loc="left", fontsize=7.5)
        ax.axhline(2e5, color=C_REF, lw=0.5, ls=":")
    axes[0].set_ylabel("pasos globales (mediana)")
    axes[1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, "comparisons", "fig_efficiency")
    pd.DataFrame(data).to_csv(SUMM / "efficiency_alpha.csv", index=False)


# ================================================================= tabla 2
def table_final_values():
    Rs, Rq = load_runs("base", "base", "sarsa"), load_runs("base", "base", "q_learning")
    tms, tmq = stack(Rs, "tail_mean"), stack(Rq, "tail_mean")
    sds = stack(Rs, "tail_sd")
    vs, vq = np.median(stack(Rs, "visits"), 0), np.median(stack(Rq, "visits"), 0)
    rows, data = [], []
    for s in range(20):
        if s not in REACH:
            continue
        cells = []
        for a in (UR, DR):
            cells += [fmt(QS[s, a]), fmt(QE[s, a]), f"{fmt(tms[:, s, a].mean())} ({sds[:, s, a].mean():.2f})",
                      fmt(tmq[:, s, a].mean()), f"{int(vs[s, a])}/{int(vq[s, a])}"]
            data.append(dict(s=s, a=a, q_star=QS[s, a], q_eps=QE[s, a], sarsa_tail_mean=tms[:, s, a].mean(),
                             sarsa_tail_sd=sds[:, s, a].mean(), ql_tail_mean=tmq[:, s, a].mean(),
                             ql_max_abs_err=np.abs(stack(Rq, "q_final")[:, s, a] - QS[s, a]).max(),
                             visits_sarsa=vs[s, a], visits_ql=vq[s, a]))
        rows.append(f"$s_{{{s}}}$ & " + " & ".join(cells) + " \\\\")
    pd.DataFrame(data).to_csv(SUMM / "final_values_base.csv", index=False)
    head = ("\\begin{tabular}{c rrrrr rrrrr}\n\\toprule\n"
            " & \\multicolumn{5}{c}{$a=\\nearrow$} & \\multicolumn{5}{c}{$a=\\searrow$} \\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
            "$s$ & $q_*$ & $q_\\varepsilon$ & SARSA ($\\sigma$) & QL & $N$ S/QL "
            "& $q_*$ & $q_\\varepsilon$ & SARSA ($\\sigma$) & QL & $N$ S/QL \\\\\n\\midrule\n")
    (TABLES / "tab_final.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n",
                                          encoding="utf-8")


# ================================================================= otras cifras
def misc_numbers():
    Rs, Rq = load_runs("base", "base", "sarsa"), load_runs("base", "base", "q_learning")
    NUMBERS["policy_stable_episode_median"] = {
        a: float(np.median([first_stable(o == len(REACH)) for o in stack(R, "ok_star")]))
        for a, R in (("sarsa", Rs), ("q_learning", Rq))}
    NUMBERS["eval_optimal_first_episode_median"] = {
        a: float(np.median([R[0]["eval_ep"][np.argmax(x >= 1.5 - 1e-9)] for x in stack(R, "eval_ret").mean(2)]))
        for a, R in (("sarsa", Rs), ("q_learning", Rq))}
    NUMBERS["steps_total_mean"] = {a: float(np.mean([r["cum_steps"][-1] for r in R]))
                                   for a, R in (("sarsa", Rs), ("q_learning", Rq))}
    NUMBERS["ql_s10_last_change_median"] = float(np.median(stack(Rq, "last_change")[:, 10]))
    NUMBERS["sarsa_s10_last_change_median"] = float(np.median(stack(Rs, "last_change")[:, 10]))
    NUMBERS["ql_final_s10_greedy_UR"] = int((stack(Rq, "greedy_final")[:, 10] == UR).sum())
    NUMBERS["sarsa_final_s10_greedy_UR"] = int((stack(Rs, "greedy_final")[:, 10] == UR).sum())
    NUMBERS["last_change_median_by_state"] = {
        a: np.median(stack(R, "last_change"), 0)[:20].tolist() for a, R in (("sarsa", Rs), ("q_learning", Rq))}
    # pasos para que Q(s,a) entre en la banda +-0.1: visitas aproximadas
    for a, R in (("sarsa", Rs), ("q_learning", Rq)):
        key = "first_eps" if a == "sarsa" else "first_star"
        vis = []
        for r in R:
            f = r[key][6, DR]
            vis.append(int(r["visits_sub"][f // 10, 6, DR]) if f >= 0 else -1)
        NUMBERS[f"{a}_visits_s6DR_at_band_approx"] = float(np.median(vis))
    for g, n in (("ext", "glie_eps"), ("ext", "glie_eps_decay_alpha")):
        R = load_runs(g, n, "sarsa")
        Rq_ = load_runs(g, n, "q_learning")
        NUMBERS[n] = dict(
            sarsa_q13ur_tail=float(stack(R, "tail_mean")[:, 13, UR].mean()),
            sarsa_q13ur_tail_sd=float(stack(R, "tail_sd")[:, 13, UR].mean()),
            sarsa_q6ur_final=float(stack(R, "q_final")[:, 6, UR].mean()),
            sarsa_ret_last10k=float(stack(R, "returns")[:, 10_000:].mean()),
            ql_ret_last10k=float(stack(Rq_, "returns")[:, 10_000:].mean()),
            sarsa_max_err_star=float(np.median(np.abs(stack(R, "q_final") - QS)[:, REACH].max(axis=(1, 2)))),
            ql_s10_dr_500_10k=float((stack(Rq_, "greedy_s10")[:, 500:10_000] == DR).mean()),
            sarsa_s10_dr_500_10k=float((stack(R, "greedy_s10")[:, 500:10_000] == DR).mean()))
    for n in ("a0.5_e0.5", "a0.05_e0.5", "e0.05_q0+5", "e0.05_q0-5"):
        NUMBERS[n] = {a: dict(eval_final=float(stack(load_runs("inter", n, a), "eval_ret")[:, -1].mean()),
                              ret_last10k=float(stack(load_runs("inter", n, a), "returns")[:, 10_000:].mean()),
                              q13ur_tail=float(stack(load_runs("inter", n, a), "tail_mean")[:, 13, UR].mean()),
                              q13ur_sd=float(stack(load_runs("inter", n, a), "tail_sd")[:, 13, UR].mean()))
                      for a in ("sarsa", "q_learning")}
    # retorno greedy en la fase inicial (episodios 20-200), comparación pareada
    es = stack(Rs, "eval_ret").mean(2)[:, 1:11].mean(1)
    eq = stack(Rq, "eval_ret").mean(2)[:, 1:11].mean(1)
    NUMBERS["eval_ret_ep20_200"] = dict(sarsa=float(es.mean()), ql=float(eq.mean()),
                                        diff_ci=bootstrap_ci(es - eq), p=sign_flip_pvalue(es - eq))
    # diferencia de retorno de entrenamiento (500-10k) por epsilon
    for e, src in {0.05: ("epsilon", "eps_0.05"), 0.1: ("epsilon", "eps_0.1"),
                   0.2: ("base", "base"), 0.5: ("epsilon", "eps_0.5")}.items():
        a_ = stack(load_runs(*src, "sarsa")[:10], "returns")[:, 500:10_000].mean(1)
        b_ = stack(load_runs(*src, "q_learning")[:10], "returns")[:, 500:10_000].mean(1)
        c_ = stack(load_runs(*src, "sarsa")[:10], "returns")[:, 10_000:].mean(1)
        d_ = stack(load_runs(*src, "q_learning")[:10], "returns")[:, 10_000:].mean(1)
        pi = QS.argmax(1).copy()
        pi[10] = UR
        th_up = expected_training_return(P, pi, e)
        pi[10] = DR
        th_dn = expected_training_return(P, pi, e)
        NUMBERS[f"train_diff_eps{e}"] = dict(
            diff_500_10k=bootstrap_ci(a_ - b_), p_500_10k=sign_flip_pvalue(a_ - b_),
            diff_10k_20k=bootstrap_ci(c_ - d_), p_10k_20k=sign_flip_pvalue(c_ - d_),
            theory_gap=th_up - th_dn, ql_s10_dr_500_10k=float(
                (stack(load_runs(*src, "q_learning")[:10], "greedy_s10")[:, 500:10_000] == DR).mean()),
            ql_s10_dr_10k_20k=float(
                (stack(load_runs(*src, "q_learning")[:10], "greedy_s10")[:, 10_000:] == DR).mean()))
    # colapso de SARSA con alpha = 0.5 (semilla 0)
    r = load_runs("alpha", "alpha_0.5", "sarsa")[0]
    q13 = r["q_track"][:, 4, UR]
    k = int(np.argmax(q13 < -5))
    NUMBERS["sarsa_alpha0.5_seed0_collapse"] = {
        "first_ep_q13ur_below_minus5": k,
        "eval_ckpts_suboptimal_frac_after_ep1000": float(
            np.mean(stack(load_runs("alpha", "alpha_0.5", "sarsa"), "eval_ret")[:, 50:].mean(2) < 1.5 - 1e-9))}
    pd.DataFrame(QE[:20]).to_csv(SUMM / "q_eps.csv")


def hyper_summary_csv():
    """Una fila por (configuración, algoritmo) con las métricas principales."""
    import run_experiments as rexp

    rows = []
    for g, cfgs in rexp.GROUPS.items():
        for c in cfgs:
            for algo in ("sarsa", "q_learning"):
                R = load_runs(g, c["name"], algo)
                er = stack(R, "eval_ret").mean(2)
                ret = stack(R, "returns")
                rows.append(dict(
                    group=g, config=c["name"], algo=algo, **{k: v for k, v in c["params"].items() if k != "rewards"},
                    seeds=len(R), ret_0_500=ret[:, :500].mean(), ret_500_10k=ret[:, 500:10_000].mean(),
                    ret_10k_20k=ret[:, 10_000:].mean(), success_10k_20k=stack(R, "success")[:, 10_000:].mean(),
                    eval_final=er[:, -1].mean(),
                    eval_first_opt_ep=np.median([R[0]["eval_ep"][np.argmax(x >= 1.5 - 1e-9)] if (x >= 1.5 - 1e-9).any()
                                                 else np.nan for x in er]),
                    policy_stable_ep=np.median([first_stable(o == len(REACH)) for o in stack(R, "ok_star")]),
                    q13_ur_tail_mean=stack(R, "tail_mean")[:, 13, UR].mean(),
                    q13_ur_tail_sd=stack(R, "tail_sd")[:, 13, UR].mean(),
                    max_err_qstar=np.median(np.abs(stack(R, "q_final") - QS)[:, REACH].max(axis=(1, 2))),
                    coverage=(stack(R, "visits")[:, REACH] > 0).sum(axis=(1, 2)).mean(),
                    s10_dr_500_10k=(stack(R, "greedy_s10")[:, 500:10_000] == DR).mean()))
    pd.DataFrame(rows).to_csv(SUMM / "hyperparameter_summary.csv", index=False)


if __name__ == "__main__":
    fig_map()
    table_qstar()
    fig_q_evolution()
    fig_alpha_s13()
    fig_returns()
    fig_epsilon()
    fig_qinit()
    fig_visits_error()
    fig_efficiency()
    table_final_values()
    misc_numbers()
    hyper_summary_csv()
    (SUMM / "numbers.json").write_text(json.dumps(NUMBERS, indent=2, default=float), encoding="utf-8")
    print("Figuras en", FIGURES, "| tablas en", TABLES, "| cifras en", SUMM)
