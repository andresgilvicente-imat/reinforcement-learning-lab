"""Lanza los experimentos de la Práctica 2 y guarda resultados reproducibles.

Uso (desde lab2/experiments):
    python run_experiments.py              # todos los grupos
    python run_experiments.py base alpha   # solo esos grupos
    python run_experiments.py --workers 8

Cada ejecución (algoritmo, configuración, semilla) usa las funciones de
episodio reales de lab2/src a través de ``common.run_training`` y guarda en
``results/raw/<grupo>/<config>/<algo>_s<semilla>.npz``:

* curvas por episodio: retorno (sin y con descuento), pasos, éxito, estado inicial;
* Q completa cada 10 episodios y Q de s3, s6, s10, s11, s13, s14 en cada episodio;
* visitas por par (s, a) cada 10 episodios y al final;
* evaluación greedy (epsilon = 0) desde los 4 estados iniciales cada 20 episodios;
* métricas por par calculadas a resolución completa (primer episodio en la
  banda |Q - ref| < 0.1, episodio a partir del cual ya no sale de ella, media
  y desviación en el último 25 % del entrenamiento) frente a q* y frente a q_eps*.

La lista de configuraciones y semillas queda en results/configurations/.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import (
    GAMMA,
    INITIAL_STATES,
    R_GOAL,
    R_OBSTACLE,
    R_STEP,
    RESULTS,
    make_env,
    optimal_action_sets,
    reachable_states,
    run_training,
    soft_optimal_q,
)

ALGOS = ("sarsa", "q_learning")
EPISODES = 20_000
BASE = {"alpha": 0.1, "epsilon": 0.2, "q_init": 0.0}
SEEDS_BASE = list(range(20))
SEEDS = list(range(10))
TRACK = (3, 6, 10, 11, 13, 14)
SUB = 10  # submuestreo de Q y visitas
TOL = 0.1  # banda para "aprendido"


def cfg(name, seeds=SEEDS, **kw):
    c = {**BASE, "episodes": EPISODES, "rewards": [R_STEP, R_OBSTACLE, R_GOAL],
         "eps_schedule": None, "alpha_schedule": None}
    c.update(kw)
    return {"name": name, "seeds": list(seeds), "params": c}


GROUPS = {
    "base": [cfg("base", SEEDS_BASE)],
    "alpha": [cfg(f"alpha_{a}", alpha=a) for a in (0.01, 0.05, 0.5, 1.0)],
    "epsilon": [cfg(f"eps_{e}", epsilon=e) for e in (0.0, 0.05, 0.1, 0.5)],
    "q_init": [cfg(f"q0_{q}", q_init=q) for q in (-5.0, 5.0)],
    "inter": [
        cfg("a0.5_e0.5", alpha=0.5, epsilon=0.5),
        cfg("a0.05_e0.5", alpha=0.05, epsilon=0.5),
        cfg("e0_q0-5", epsilon=0.0, q_init=-5.0),
        cfg("e0_q0+5", epsilon=0.0, q_init=5.0),
        cfg("e0.05_q0+5", epsilon=0.05, q_init=5.0),
        cfg("e0.05_q0-5", epsilon=0.05, q_init=-5.0),
    ],
    "ext": [
        # GLIE: eps_k = 0.2 / (1 + k/1000) -> 0.0095 al final
        cfg("glie_eps", eps_schedule="glie"),
        # GLIE + alpha_k = 0.1 (1 + k/1000)^-0.6  (Robbins-Monro en k)
        cfg("glie_eps_decay_alpha", eps_schedule="glie", alpha_schedule="rm"),
    ],
}


def schedule(name, base):
    if name is None:
        return None
    if name == "glie":
        return lambda k: base / (1 + k / 1000)
    if name == "rm":
        return lambda k: base * (1 + k / 1000) ** -0.6
    raise ValueError(name)


def band_metrics(q, ref, tol=TOL):
    """q: (E, S, A). Primer episodio en la banda y episodio desde el que no sale."""
    inside = np.abs(q - ref[None]) < tol
    E = q.shape[0]
    first = np.where(inside.any(0), inside.argmax(0), -1)
    outside = ~inside
    last_out = np.where(outside.any(0), E - 1 - outside[::-1].argmax(0), -1)
    settle = np.where(inside[-1], last_out + 1, -1)
    return first.astype(np.int32), settle.astype(np.int32)


def worker(group, name, params, algo, seed):
    p = params
    P = make_env(*p["rewards"]).unwrapped.P
    eps_ref = 0.0 if p["eps_schedule"] else p["epsilon"]
    qstar = soft_optimal_q(P, GAMMA, 0.0)[0]
    qeps = soft_optimal_q(P, GAMMA, eps_ref)[0]
    t0 = time.time()
    r = run_training(
        algo, p["episodes"], p["alpha"], p["epsilon"], p["q_init"], seed,
        rewards=tuple(p["rewards"]),
        eps_schedule=schedule(p["eps_schedule"], p["epsilon"]),
        alpha_schedule=schedule(p["alpha_schedule"], p["alpha"]),
    )
    qh = r["q_hist"]
    E = qh.shape[0]
    tail = qh[int(0.75 * E):]
    reach = reachable_states(P)
    a_star = optimal_action_sets(qstar)
    a_eps = optimal_action_sets(qeps)
    greedy = qh.argmax(2)  # (E, S)
    ok_star = np.array([[greedy[k, s] in a_star[s] for s in reach] for k in range(0, E)]).sum(1)
    ok_eps = np.array([[greedy[k, s] in a_eps[s] for s in reach] for k in range(0, E)]).sum(1)
    changes = np.vstack([np.zeros((1, qh.shape[1]), bool), greedy[1:] != greedy[:-1]])
    last_change = np.where(changes.any(0), E - 1 - changes[::-1].argmax(0), -1)
    out = {}
    for tag, ref in (("star", qstar), ("eps", qeps)):
        first, settle = band_metrics(qh, ref)
        out[f"first_{tag}"], out[f"settle_{tag}"] = first, settle
    out.update(
        q_sub=qh[SUB - 1 :: SUB].astype(np.float32),
        q_track=qh[:, TRACK].astype(np.float32),
        q_final=r["Q"],
        tail_mean=tail.mean(0), tail_sd=tail.std(0), tail_min=tail.min(0), tail_max=tail.max(0),
        visits=r["visits"],
        visits_sub=r["visits_hist"][SUB - 1 :: SUB],
        returns=r["returns"].astype(np.float32),
        disc_returns=r["disc_returns"].astype(np.float32),
        steps=r["steps"].astype(np.int16),
        success=r["success"],
        starts=r["starts"],
        eval_ep=r["eval_ep"], eval_ret=r["eval_ret"].astype(np.float32), eval_succ=r["eval_succ"],
        ok_star=ok_star.astype(np.int8), ok_eps=ok_eps.astype(np.int8),
        greedy_final=greedy[-1].astype(np.int8), last_change=last_change.astype(np.int32),
        greedy_s10=greedy[:, 10].astype(np.int8),
        qstar=qstar, qeps=qeps, track=np.array(TRACK),
    )
    dest = RESULTS / "raw" / group / name / f"{algo}_s{seed}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    return f"{group}/{name}/{algo}/s{seed}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    conf_dir = RESULTS / "configurations"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "experiments.json").write_text(
        json.dumps({"gamma": GAMMA, "initial_states": list(INITIAL_STATES), "algorithms": ALGOS,
                    "tolerance_band": TOL, "groups": GROUPS}, indent=2), encoding="utf-8")

    jobs = [(g, c["name"], c["params"], a, s)
            for g in args.groups for c in GROUPS[g] for a in ALGOS for s in c["seeds"]]
    print(f"{len(jobs)} entrenamientos con {args.workers} procesos")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(worker, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            tag, dt = f.result()
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {tag} ({dt:.1f}s) total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
