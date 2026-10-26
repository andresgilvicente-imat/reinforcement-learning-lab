"""Carga de los resultados guardados por run_experiments.py."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from common import RESULTS


@lru_cache(maxsize=None)
def load_runs(group, name, algo):
    """Lista de dicts (uno por semilla, ordenados por semilla)."""
    folder = RESULTS / "raw" / group / name
    files = sorted(folder.glob(f"{algo}_s*.npz"), key=lambda f: int(f.stem.split("_s")[-1]))
    if not files:
        raise FileNotFoundError(f"{folder}/{algo}_s*.npz")
    runs = []
    for f in files:
        with np.load(f) as d:
            run = {k: d[k] for k in d.files}
        run["seed"] = int(f.stem.split("_s")[-1])
        run["cum_steps"] = np.cumsum(run["steps"].astype(np.int64))
        runs.append(run)
    return runs


def stack(runs, key):
    return np.stack([r[key] for r in runs])


def ep_to_steps(run, ep):
    """Episodio (índice 0..E-1, -1 = nunca) -> pasos globales acumulados."""
    ep = np.asarray(ep)
    out = np.where(ep >= 0, run["cum_steps"][np.clip(ep, 0, None)], -1)
    return out


def first_stable(mask):
    """Primer índice a partir del cual ``mask`` es siempre True (-1 si no)."""
    mask = np.asarray(mask, bool)
    if not mask[-1]:
        return -1
    bad = np.flatnonzero(~mask)
    return int(bad[-1] + 1) if bad.size else 0
