# curve-continuation

Curve continuation across fragment borders, used as a compatibility for fresco and puzzle reassembly.

Test pictures are drawn from curve equations and cut into pieces, so the true layout and the exact curve ends are
known. The curve ends are found in the piece images, every placement of two pieces is scored by how smoothly the
curves continue across the gap (biarc, co-circularity or clothoid, plus width, colour and curvature), and the group's
RL solver assembles the puzzle from these scores. The solver is used as it is, never modified.

## Install

Python 3.10 or newer.

```
pip install -e ".[solver,dev]"
```

`solver` adds what the group's solver imports (OpenCV, shapely, matplotlib, natsort, scikit-fmm, sympy) and `dev` adds
pytest and ruff; if the solver already runs on your computer, `pip install -e ".[dev]"` is enough. The scripts also
run without installing, from the project folder.

Put the group's solver next to the code, as `RL_puzzle_solver/new_code` inside this folder, or write its path in
`configs/default.yaml` under `solver: folder:` (an absolute path is fine). Puzzles are written to its `data/` folder.

## Run

One picture through the whole pipeline (draw, cut, find the ends, compare them with the exact ends, solve):

```
python scripts/run_pipeline.py
```

```
1. .../data/preprocessing/circle_3x3_00000: 9 pieces of 185 x 185 px, gap 15 px, 16 exact curve ends ...
2. 9 pieces (1 without a curve end), 16 curve ends, 16 fitted, 0 failed
3. circle_3x3_00000: 16 exact ends, 16 found by extraction, 0 missed, 0 extra
4. circle_3x3_00000: score biarc_lap, pieces in the right place 9 / 9 ...
```

Each thesis experiment is one file in `configs/experiments/`. For each one:

```
python scripts/run_dataset.py configs/experiments/frescoes_lap.yaml    # makes the pictures, finds the ends, solves
python scripts/compare.py configs/experiments/frescoes_lap.yaml        # solver: two methods x found / exact ends
python scripts/check_pairs.py configs/experiments/frescoes_lap.yaml    # right / wrong pairs, no solver
```

`compare.py` and `check_pairs.py` use the newest puzzle of each picture, so `run_dataset.py` has to run first, once
per dataset (experiments on the same dataset share the puzzles). Other useful forms:

```
python scripts/run_pipeline.py configs/experiments/frescoes_lap.yaml --picture fresco_03
python scripts/compare.py configs/experiments/colour_nfa.yaml --methods biarc clothoid --ends curves
python scripts/make_frescoes.py --n 50 --seed 7 --out datasets/frescoes_50.yaml
```

### Expected results

Pieces in the right place, total over the dataset. `run_dataset.py` uses biarc with found ends, which is the first
column of `compare.py`.

| experiment | what it shows | compare.py: found ends | exact ends |
|---|---|---|---|
| `canvases_baseline` | one closed curve per picture, best single join per side | biarc 71/90, co-circ 70/90 | 80/90, 77/90 |
| `multi_lap` | several curves per picture, one-to-one matching (LAP) | biarc 46/54, co-circ 46/54 | 52/54, 46/54 |
| `frescoes_lap` | fresco-like pictures, LAP | biarc 148/180, co-circ 126/180 | 177/180, 150/180 |
| `frescoes_order` | matching that keeps the order along the side, bad joins refused | biarc 151/180, co-circ 126/180 | 176/180, 148/180 |
| `colour_lap` | fresco colours, geometry only | biarc 145/180, co-circ 124/180 | 177/180, 150/180 |
| `colour_features` | plus width 2, colour 1, curvature 0.5 | biarc 149/180, co-circ 138/180 | 178/180, 175/180 |
| `colour_clothoid` | plus curvature checked against the clothoid | biarc 150/180, clothoid 152/180 | 178/180, 178/180 |
| `colour_nfa` | all features plus the a contrario test | biarc 152/180, clothoid 154/180 | 178/180, 177/180 |

`check_pairs.py configs/experiments/colour_features.yaml` gives 364 right, 36 wrong, 114 negative pairs for biarc
with found ends. Its solver column counts the largest group of pieces that are right together, whatever the shift of
the whole layout, so it can be a little higher than `compare.py`, which counts relative to the anchor piece. These are the numbers of the scripts before the restructure; the package reproduces them exactly
(same pictures, same ends file for file, same 64 pair counts in `tests/data/reference.json`, same solver totals).

## Settings

`configs/default.yaml` holds every setting with a comment on what it does, in groups: `solver`, `picture`, `curve`
(the picture of `run_pipeline.py`), `extraction`, and `score` (`geometry`, `rules`, `matching`, `features`,
`significance`). An experiment file lists only what differs:

```yaml
description: fresco colours, join cost C = C_geometry + 2 C_width + 1 C_colour + 0.5 C_curvature
dataset: datasets/frescoes_colour.yaml
compare:
  methods: [biarc, cocircularity]   # the columns of compare.py and check_pairs.py
  ends: [curves, exact]
settings:
  score:
    features:
      width: 2.0
      colour: 1.0
      curvature: 0.5
```

A misspelt setting or an impossible value stops the run with a message, so a typo cannot change a result silently.
Every run writes `results/<experiment>/<date_time>_<script>/` with the complete settings (`config.yaml`), the
command and the code version (`run.txt`), and its tables (`table.txt`, CSV files). Keep the project in git
(`git init` once, then commit as you go): `run.txt` then names the commit of the code that made each result. The solver keeps its solutions in
`RL_puzzle_solver/new_code/data/experiments/<puzzle>/`, in folders named after the score settings, such as
`biarc_lap_sol_...` or `clothoid_exact_lap_w2c1k0.5_nfa_sol_...` (w, c, k = width, colour, curvature weights).

### The score in short

- Join cost of two ends: `C = C_geometry + w_width q C_width + w_colour q C_colour + w_curvature q C_curvature`, where
  `q = 0` for a measurement that extraction marked unreliable (merged strokes, poor curvature fit).
- Join score: `s = clip(1 - C / tau, -1, 1)`.
- Placement score `S`: overlapping pieces -1, more than `max_gap` apart 0, diagonal neighbours the best join near the
  shared corner, side by side the ends of the two facing edges combined by the matching (`best`, `lap`, `order`);
  an end without a partner counts `unmatched_score`, or 0 near a corner.
- With `significance: nfa`, a placement keeps its support only when its joins are unlikely between unrelated ends
  (number of false alarms below `epsilon`).

## Folders

```
configs/default.yaml       all settings, with their meaning
configs/experiments/       one file per experiment: dataset + what differs from the defaults
datasets/                  test pictures as curve equations (canvases, multi_curves, frescoes, frescoes_colour)
src/curvecont/
    config.py              loading and checking settings; ScoreParams, the flat score settings
    synth/                 curves.py: draw from equations, cut, exact ends; frescoes.py: fresco-like pictures
    extraction/ends.py     find curve ends (position, direction, curvature, width, colour) in piece images
    scoring/               ends.py, geometry.py (biarc, co-circularity, clothoid), features.py (width, colour,
                           curvature), matching.py (best, lap, order), significance.py (a contrario test),
                           compatibility.py (S of a placement, the matrix R for the solver)
    evaluation/            ends.py (found vs exact ends), pairs.py (right / wrong pairs),
                           placement.py (pieces in the right place)
    puzzles.py             reading puzzles and dataset files
    pipeline.py            one picture: make, extract, compare, solve
    solver_bridge.py       runs the group's solver with our R, in a process of its own
scripts/                   command-line entry points: read the arguments, call the package, print
tests/                     pytest; tests/data/reference.json holds the numbers of the original scripts
results/                   written by the scripts (not in git)
thesis/                    text and figures
legacy/                    early stand-alone tools, not used by the pipeline
RL_puzzle_solver/          the group's solver, its own repository (not in git)
```

## Tests

```
pytest                  # everything, about a minute and a half
pytest -m "not slow"    # without the 64 pair counts of the original scripts, about 20 s
ruff check .
```

## From the previous scripts

| before | now |
|---|---|
| `python src/run_all.py settings.yaml` | `python scripts/run_pipeline.py [experiment]` |
| `python src/run_canvases.py frescoes.yaml` | `python scripts/run_dataset.py configs/experiments/frescoes_lap.yaml` |
| `python src/compare_scores.py frescoes.yaml --methods ...` | `python scripts/compare.py <experiment> [--methods ...]` |
| `python src/check_scores.py frescoes.yaml` | `python scripts/check_pairs.py <experiment>` |
| `python src/make_frescoes.py --colour` | `python scripts/make_frescoes.py --colour` |
| one `settings.yaml`, edited between runs | `configs/default.yaml` plus one file per experiment |
| `score: method, tau` | `score: geometry: method, tau` |
| `score: unmatched_score, max_gap, corners` | `score: rules: ...` |
| `score: matching, skip_bad_joins` | `score: matching: method, skip_bad_joins` |
| `lambda_width, lambda_colour, lambda_curvature, curvature_check` | `score: features: width, colour, curvature, curvature_check` |
| `significance, epsilon, nfa_tests` | `score: significance: mode, epsilon, tests` |
| extraction options on the command line of `extract.py` | `extraction:` in the settings |
| datasets with `canvases:` | `pictures:` (`canvases:` is still read) |

## Troubleshooting

- `solver.folder is not the new_code folder of RL_puzzle_solver`: set `solver: folder:` in `configs/default.yaml`.
- `Memory allocation still failed` from OpenBLAS: the scripts already use one OpenBLAS thread; close other programs.
- A cell of `compare.py` shows `failed`: the solver stopped on that puzzle (it can fail in
  `remove_occupied_grid_points`); the message is in `errors.txt` in the results folder.
