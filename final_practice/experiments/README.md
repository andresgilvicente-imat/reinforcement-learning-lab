# Hyperparameter experiments (Práctica 3 · Sesión 1)

Tools to choose the hyperparameters of the Sarsa and Q-learning agents for the
leaderboard **without overfitting** to any particular set of start positions,
and to produce the tables and figures for the final presentation.

```
experiments/
├── barrido.py        # trains many configurations in parallel and evaluates them
├── analiza.py        # summarises the CSVs (tables + heatmap figure)
├── README.md
└── resultados/       # one CSV per sweep (+ PNG heatmaps)
```

Nothing in `src/` is modified. The scripts *import* the practice code
(`entrenamiento.entrena`, `episodio_sarsa`, `episodio_q_learning`, your
`recompensa`, `evaluacion.evalua`...), so whatever you change in your gap files
is what gets tested.

---

## 1. Quick start

Requirements: the same environment as the practice (`numpy gymnasium matplotlib`).
Run everything from the `final_practice/` folder.

```bash
# Stage 1: coarse grid, 3 seeds per configuration
python experiments/barrido.py --algoritmo sarsa \
    --rho 0.08 0.16 0.24 0.4 --n_tiles 6 8 10 12 16 --num_tilings 4 8 16 \
    --seeds 0 1 2 --salida experiments/resultados/e1_sarsa.csv

# Summary ranked for leaderboard 1 (arrivals) + heatmap PNG
python experiments/analiza.py experiments/resultados/e1_sarsa.csv --top 15 --figura

# Summary ranked for leaderboard 2 (extra steps; only configs with >= 500 arrivals)
python experiments/analiza.py experiments/resultados/e1_sarsa.csv --orden extra
```

Approximate times on 14 processes: Sarsa ~1 s per training, Q-learning ~2 s
(180 trainings ≈ 3–5 min).

---

## 2. What one experiment does

A **configuration** is

```
algorithm · ρ · tiles per dimension · number of tilings · q0 · ε schedule · seed
```

For each configuration `barrido.py`:

1. **Builds the agent** exactly like `main.py`:
   `TileCoder((n, n), tilings)` → `AgenteLineal(coder, q0)`, with
   `alpha = ρ / num_tilings`.
2. **Trains** it with `entrenamiento.entrena(...)`, the same function `main.py`
   calls, with your `episodio_*` function, 2000 episodes, γ = 0.99 and the
   given seed. Because the call is identical, **the weights are bit-for-bit the
   same as `python main.py --seed <seed> ...`** (checked: same 95/100 arrivals
   and 25.47 mean steps as `main.py` for the same settings).
3. **Evaluates** the greedy policy with `evaluacion.evalua`, the teachers' own
   evaluation function (200 steps max), on the *selection* positions.
4. **Writes one CSV row**: arrivals, crashes, mean extra steps, max |w|,
   whether it diverged, training steps, time.

### Why ρ instead of α

In every state exactly one tile per tiling is active, so one update moves q̂ by
`α · num_tilings · δ`. Sweeping the **effective step ρ = α · num_tilings** makes
designs with a different number of tilings comparable: 4 tilings with α = 0.1
and 16 tilings with α = 0.025 take the same size of step. `analiza.py` also
prints the corresponding α, which is the value you pass to `main.py --alpha`.

### Leaderboard 2: extra steps over the shortest path

`camino_minimo(x, y)` runs a **BFS** from the start position over the grid the
agent can really move on (±0.25 m in x or y), using the environment's own
`is_collision` and `in_target`. The first cell inside the target gives the
exact minimum number of steps. Extra steps = agent steps − minimum, averaged
over the episodes that arrive (the same definition as the leaderboard).
Sanity checks: no agent ever goes below the minimum, and the longest shortest
path is 55 steps (`evaluacion.py` says "no start needs more than 60").

### Divergence detection

Each episode is wrapped by `_vigilado`: if any |w| exceeds `W_DIVERGE = 1e4`
(meaningful q̂ values lie roughly in [−100, 100]) training stops and the row is
stored with `diverge = 1` and 0 arrivals. Without this, a diverged Q-learning
agent would loop through 2000-step episodes and waste minutes. Note that
`entrena` itself only catches `inf`/`nan`, so weights of 10⁷⁴ would pass
silently.

---

## 3. How the anti-overfitting mechanism works

### The problem

The teachers evaluate **once**, on **1000 start positions we never see**. Those
positions are drawn from the same distribution as `env.reset()` (uniform over
the free area). So what we really want to maximise is the **expected**
performance over that distribution, not the score on any concrete set of
positions.

Every number we measure is that expectation **plus noise**, and the noise has
two sources:

| Source of noise | Why it exists |
|---|---|
| **Which positions** we evaluate on | Some starts are easy (next to the target), some hard (behind a wall, near a corner). A small set can be lucky or unlucky. |
| **Which training seed** we use | The seed fixes start positions, exploratory actions… during training. The same hyperparameters can give a good or a mediocre agent depending on it. |

**Overfitting = choosing something because of the noise, not because it is
really better.** If we try many configurations and pick the top score, we tend
to pick the one that got lucky ("winner's curse"), and the real leaderboard
result will be worse than what we measured.

### The three countermeasures

**1. Large, independent selection set (fights position noise).**
We do *not* choose using the 100 validation positions that `main.py` prints.
`barrido.py` evaluates on `N_SEL = 1000` positions generated with their own
seed (`SEMILLA_SEL = 777`). The standard error of a success rate p measured on
n positions is √(p(1−p)/n):

| true success | n = 100 | n = 1000 |
|---|---|---|
| 98 % | ±1.4 % | ±0.44 % |
| 90 % | ±3.0 % | ±0.95 % |

With 100 positions, two agents that differ by 2–3 % are indistinguishable;
with 1000 they are not. It also has the same size as the teachers' bank.

**2. Several training seeds, choose by the average (fights seed noise).**
Every configuration is trained with 3 seeds in stage 1 and 5 in stage 2.
`analiza.py` ranks by the **mean** across seeds and also shows the **worst
seed** (`min`) and the **standard deviation**. A configuration with a high mean
and a high worst case is robust: it does not depend on a lucky seed. In stage 1
we saw seed standard deviations of up to 30–40 arrivals, so a single run per
configuration would have been very misleading.

**3. A separate test set, used only at the end (honest estimate).**
Even with 1–2, selecting the best of many configurations on the selection set
gives a slightly optimistic number. So the final agents are evaluated on a
third, untouched set of positions (`--test` → `SEMILLA_TEST = 4242`). Because
nothing was chosen on it, its result is an unbiased estimate of what to expect
on the leaderboard. **Use it only for the final agents.** If you start
comparing configurations on the test set and pick the best, it becomes just
another selection set and loses its value.

### The three sets of positions

| Set | Seed | Size | Used for |
|---|---|---|---|
| Validation (from `main.py`) | 2627 | 100 | Watching learning curves while you play. **Not for choosing.** |
| Selection (`barrido.py`) | 777 | 1000 | Ranking configurations. |
| Test (`barrido.py --test`) | 4242 | 1000 | Final estimate of the chosen agents, once. |
| Leaderboard | unknown | 1000 | The teachers. |

All of them come from the same distribution, so a configuration that is good
on average across seeds on the selection set will be good on the leaderboard.

### What about choosing the seed of the submitted agent?

The leaderboard retrains with *your* seed, so the seed is effectively a
hyperparameter. Picking the best of the 3–5 seeds of the chosen configuration
on the selection set is acceptable (the gain is small and real for that seed),
but:

- choose the **configuration** first, by the mean; pick the seed only among the
  seeds of that configuration;
- the estimate you report is the **test-set** score of that exact agent, not
  the selection score.

### A bias to keep in mind for leaderboard 2

Extra steps are averaged **only over episodes that arrive**. An agent that
fails the hard starts (far away, behind walls) is averaged over easier
episodes, so it looks better than it is. That is why `--orden extra` only keeps
configurations whose worst seed reaches ≥ 500 arrivals (the leaderboard rule),
and why you should look at extra steps **together with** arrivals.

---

## 4. Re-running after you change the code

The CSVs are only valid for the code that produced them. **If you change
`recompensa.py`, `episodio_*`, or the ε functions, old results are stale.**
Write new sweeps to new files (the scripts *append* to an existing CSV, so
reusing a name mixes old and new rows).

### Changing the ε schedule

There are two ways to test schedules:

- **Without editing the code**: `--eps ini,fin,f1,f2` uses the same shape as
  the current `epsilon_sarsa` (constant `ini` until `f1·N`, exponential decay
  to `fin` until `f2·N`, then constant `fin`; with `fin = 0` the decay is
  linear down to 0). Several can be compared in one run:

  ```bash
  python experiments/barrido.py --algoritmo sarsa --rho 0.4 --n_tiles 10 \
      --num_tilings 4 --eps codigo 0.7,0,0.2,0.85 0.5,0.001,0.1,0.8 \
      --seeds 0 1 2 3 4 --salida experiments/resultados/e2_sarsa_eps.csv
  ```

- **`--eps codigo`** (the default) uses whatever is currently written in
  `epsilon_sarsa` / `epsilon_q_learning`. If you change the *shape* of the
  schedule (e.g. a cosine or 1/k decay), edit the code and run with
  `codigo`, or add the new shape to `ProgramaEpsilon` in `barrido.py`.

### Changing the reward

`--recompensa` chooses the reward used for training (evaluation never uses it:
`evaluacion.py` runs with a null reward, so the leaderboard only sees arrivals
and steps):

| Value | Reward | Use |
|---|---|---|
| `src` (default) | whatever `src/recompensa.py` contains **now** (what gets submitted) | final checks |
| `base` | `experiments/recompensa_base.py`: +100 goal, −100 crash, −1 step (the original one) | reproducing e1–e4 |
| `shaping` | `experiments/recompensa_shaping.py`: base + potential-based shaping | e5–e6 |

`base` and `shaping` are frozen copies, so old experiments stay reproducible
even if `src/recompensa.py` changes. To test a new reward idea, add another
`recompensa_<name>.py` and a branch in `recompensa_fn`; only when it wins, copy
it into the gap of `src/recompensa.py` and check with `--recompensa src` that it
reproduces the rows **seed by seed**.

Note: in e1–e3 the column is missing (old CSVs); the analysis scripts read it
as `base`.

### Potential-based shaping (what `recompensa_shaping.py` / `src/recompensa.py` do)

`r'(s, a, s') = r(s, a, s') + γ·Φ(s') − Φ(s)`, with `Φ(terminal) = 0` and
`Φ(s) = −(geodesic distance from s to the target, in steps)`, computed once by
a BFS on a 5 cm grid of the free space (built on the first call and cached).

- **It does not change the optimal policy** (Ng, Harada & Russell, 1999): along
  any episode the shaping terms telescope to `−Φ(s₀)`, the same constant for
  every policy.
- **It makes the signal dense**: a step towards the goal gets ≈ 0, a step away
  ≈ −2, instead of −1 both ways, so value information does not have to be
  propagated back from the goal episode by episode.
- **It removes Q-learning's divergence**: with the base reward Q-learning
  diverged from ρ ≈ 0.06 and had to use 6×6 tiles; with shaping, 0 divergences
  in 270 runs up to ρ = 0.1 and 16×16 tiles. Targets no longer depend on long
  chains of bootstrapped `max` values, which weakens the deadly triad.
- `GAMMA = 0.99` is written inside the function: it must match `main.py --gamma`
  for the invariance to hold.

### Useful options of `barrido.py`

| Option | Meaning |
|---|---|
| `--algoritmo sarsa\|q_learning` | algorithm (one per run) |
| `--rho …` | effective step sizes ρ = α·tilings |
| `--n_tiles …` | tiles per dimension: `12` (12×12) or `12x16` (x by y) |
| `--recompensa …` | `src`, `base` and/or `shaping` (see above) |
| `--num_tilings …` | number of tilings |
| `--q0 …` | initial value of q̂ (optimistic initialisation) |
| `--eps …` | `codigo` and/or `ini,fin,f1,f2` |
| `--seeds …` | training seeds |
| `--n_sel` | number of evaluation positions (default 1000) |
| `--test` | evaluate on the test positions (final agents only) |
| `--procesos` | parallel processes (default: CPUs − 2) |
| `--salida` | CSV file (rows are appended) |

All list options are combined as a full grid (Cartesian product). To test
specific, non-grid combinations, call it once per combination with the same
`--salida` (see stage 2 below).

---

## 5. Recommended workflow

1. **Stage 1, coarse grid** of ρ × tiles × tilings, 3 seeds, `--eps codigo`.
   Look at the heatmaps: where the good region is, where Q-learning diverges.
2. **Stage 2, refine**: the 4–6 best designs × several ε schedules (and maybe
   `q0`), 5 seeds:

   ```bash
   for c in "0.4 10 4" "0.16 8 4" "0.24 12 8"; do set -- $c
     python experiments/barrido.py --algoritmo sarsa --rho $1 --n_tiles $2 \
         --num_tilings $3 --eps codigo 0.7,0,0.2,0.85 0.5,0.001,0.1,0.8 \
         --seeds 0 1 2 3 4 --salida experiments/resultados/e2_sarsa.csv
   done
   python experiments/analiza.py experiments/resultados/e2_sarsa.csv
   ```
3. **Choose** one configuration per algorithm by the **mean** (and worst case).
   Since the group scores with its best agent in *each* leaderboard, one agent
   can target arrivals and the other extra steps.
4. **Write the chosen ε schedule into `sarsa.py` / `q_learning.py`** and verify
   with `--eps codigo` that you get the same numbers.
5. **Test once** on the test set:

   ```bash
   python experiments/barrido.py --algoritmo sarsa --rho 0.4 --n_tiles 10 \
       --num_tilings 4 --seeds 3 --test --salida experiments/resultados/final.csv
   ```
6. **Generate the submission** (each algorithm separately, because `ambos`
   shares α and the tiling; both runs use the same prefix):

   ```bash
   cd src
   python main.py --algoritmo sarsa      --alpha <α_s> --n_tiles <n> <n> \
       --num_tilings <t> --seed <s> --guarda G7 --grupo G7
   python main.py --algoritmo q_learning --alpha <α_q> --n_tiles <n> <n> \
       --num_tilings <t> --seed <s> --guarda G7 --grupo G7
   python evalua_agente.py G7_sarsa.npz --n 1000 --seed 4242   # must match step 5
   python evalua_agente.py G7_q_learning.npz --n 1000 --seed 4242
   ```

   The `.npz` stores the weights, `n_tiles`, `num_tilings`, `alpha`, `gamma`,
   `q0` and `seed`; the teachers retrain with these and your submitted code.
   **Do not touch the code after saving the weights**, or they will not
   reproduce and the agent is disqualified.

---

## 6. Files in `resultados/`

| File | Stage | What varies | Fixed | Seeds |
|---|---|---|---|---|
| `e1_<alg>.csv` | 1 · tile coding and step | ρ × tiles (6–16) × tilings (4, 8, 16) | ε = code at that time (0.7 → 0.02) | 0–2 |
| `e2_<alg>.csv` | 2 · ε schedule | 5 schedules on the 6 best e1 designs | design | 0–4 |
| `e3_<alg>.csv` | 3 · less exploration | 4 low-exploration schedules on the 4 best designs | design | 0–4 |
| `e4_<alg>.csv` | 4 · non-square tiles (finer in y), base reward | tiles `10x16`…`16x20` × ρ × tilings | ε = 0.2 → 0.001 | 0–4 |
| `e5_<alg>.csv` | 5 · reward shaping | ρ × square tiles (+ `12x16`) × tilings | ε = 0.2 → 0.001 | 0–4 |
| `e6_<alg>.csv` | 6 · shaping **and** non-square tiles | ρ × tiles (`10x16`, `12x16`, `12x20`, `16x20`, 14, 16) × tilings | ε = 0.2 → 0.001 | 0–4 |
| `final_test_v1_base.csv` | final check, **test** positions, first champions (base reward) | 2 agents | | 2 |
| `final_test_v2_shaping.csv` | final check, **test** positions, current champions (shaping) | 2 agents | | 1, 2 |
| `e1_<alg>.png` | heatmap: arrivals by tiles × tilings, one panel per ρ | | | |
| `e2_<alg>+e3_eps.png` | heatmap: design × ε schedule, arrivals and extra steps (divergences marked) | | | |
| `e2_<alg>+e3_programas.png` | the shape of every ε schedule tested | | | |
| `pareto.png` | arrivals vs extra steps, all experiments, Pareto front and chosen config | | | |

In the `eps` column, `codigo` means *whatever `epsilon_*` contained when that
file was run* (0.7 → 0.02 in e1/e2, 0.2 → 0.001 afterwards);
`(ini, fin, f1, f2)` is a schedule passed by the sweep. Likewise
`recompensa = src` means *whatever `src/recompensa.py` contained* (shaping, in
`final_test_v2_shaping.csv`).

### What each stage taught us

| Stage | Result |
|---|---|
| e1 | Coarse tiles + few tilings learn fastest in 2000 episodes; fine tiles give more precise paths. Q-learning needs ρ ≈ 10× smaller than Sarsa and diverges with coarse tiles (deadly triad). |
| e2–e3 | Low exploration is best for both. Sarsa: ε → 0 removes the "safety margin" (extra steps 0.53 → 0.07). Q-learning: more exploration ⇒ more divergence (behaviour policy further from the greedy target). |
| e4 | Non-square tiles alone: +arrivals within noise but many more extra steps → no gain. |
| e5 | Shaping: Q-learning 989 → 998.6 arrivals, no divergences, can use 12×12 tiles. Sarsa within noise. |
| e6 | Shaping **+** finer tiles: both reach ≈ 999.4–999.6 / 1000. Shaping is what unlocks the finer resolution (with the base reward, finer tiles did not help). Among the fine designs, the non-square ones get the **same arrivals with more precise paths** (Sarsa 0.02 vs 0.08 extra steps with 16×16; Q-learning 0.06 vs 0.14): resolution where it is needed (y), fewer weights to learn. |

Remaining failures of the first champions were all starts at y ≈ 0.6–0.8,
the 0.6 m-wide passage under the obstacles, which a 0.83 m tile cannot
resolve. Finer y resolution + a dense reward fix exactly that.

`analiza.py --figura` chooses the figure automatically: tiles × tilings when
only the design varies (stage 1), design × ε when several schedules are present
(stages 2–3). Pass several CSVs to combine stages:

```bash
python experiments/analiza.py experiments/resultados/e2_sarsa.csv \
    experiments/resultados/e3_sarsa.csv --figura
```

### Choosing with both leaderboards: `elige.py`

```bash
python experiments/elige.py --figura          # all e*_*.csv, both algorithms -> resultados/pareto.png
```

1. Keeps valid configurations (no divergence, worst seed ≥ 500 arrivals).
2. Builds the **Pareto front**: configurations that no other one beats in
   arrivals *and* extra steps at the same time.
3. Picks from the front: every configuration whose mean arrivals are within
   `2 · standard error` of the best counts as *tied* (the difference may be
   noise); among the tied ones it takes the **fewest extra steps** (that
   difference is real). `--tol` overrides the tolerance.
4. Repeats the rule among the seeds of the chosen configuration and prints the
   `main.py` command. Only selection rows are used; duplicated
   (configuration, seed) rows across files are counted once.

### Chosen agents

Current champions (reward with shaping in `src/recompensa.py`, ε = 0.2 → 0.001,
[0.05, 0.6] in `sarsa.py` / `q_learning.py`):

| Agent | α | tiles × tilings | seed | Selection (mean of 5 seeds) | Selection (this seed) | **Test** |
|---|---|---|---|---|---|---|
| Sarsa | 0.04 | 12×20 × 4 | 1 | 999.4 arr., 0.02 extra | 999, 0.01 | **1000 arr., 0.01 extra** |
| Q-learning | 0.01 | 12×16 × 8 | 2 | 999.4 arr., 0.06 extra | 1000, 0.02 | **1000 arr., 0.03 extra** |

```bash
cd src
python main.py --algoritmo sarsa      --alpha 0.04 --n_tiles 12 20 --num_tilings 4 --seed 1 --guarda G7 --grupo G7
python main.py --algoritmo q_learning --alpha 0.01 --n_tiles 12 16 --num_tilings 8 --seed 2 --guarda G7 --grupo G7
python evalua_agente.py G7_sarsa.npz      --n 1000 --seed 4242   # expect 1000, 0.01 extra
python evalua_agente.py G7_q_learning.npz --n 1000 --seed 4242   # expect 1000, 0.03 extra
```

Previous champions (base reward), kept for the presentation: Sarsa α 0.06,
12×12 × 4, seed 2 → test 998 / 0.05; Q-learning α 0.01, 6×6 × 4, seed 2 →
test 999 / 0.15 (`final_test_v1_base.csv`).

---

## 7. CSV columns

| Column | Meaning |
|---|---|
| `algoritmo, rho, n_tiles, num_tilings, q0, eps, seed` | configuration |
| `alpha` | ρ / num_tilings (value for `--alpha`) |
| `diverge` | 1 if \|w\| exceeded 1e4 during training |
| `llegadas`, `tasa` | arrivals out of `n_sel`, and the rate |
| `choques` | episodes that ended in a crash (the rest of the failures loop until 200 steps) |
| `extra` | mean extra steps over the BFS minimum, among arrivals |
| `pasos_entreno` | total environment steps used in training (sample efficiency) |
| `wmax` | max \|w\| at the end of training |
| `segundos` | training + evaluation time |
| `conjunto` | `sel` or `test` |
