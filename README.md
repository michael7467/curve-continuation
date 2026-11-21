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

Put the group's solver inside this folder as `RL_puzzle_solver/new_code`, or anywhere else and write its path in
`configs/default.yaml` under `solver: folder:`. A relative path is read from the project folder, so
`../RL_puzzle_solver/new_code` is a solver folder next to the project; an absolute path works too. Puzzles are written
to the solver's `data/` folder.

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
per dataset (experiments on the same dataset share the puzzles). The solver column of `check_pairs.py` counts the
largest group of pieces that are right together, whatever the shift of the whole layout, so it can be a little higher
than `compare.py`, which counts relative to the anchor piece. Other useful forms:

```
python scripts/run_pipeline.py configs/experiments/frescoes_lap.yaml --picture fresco_03
python scripts/compare.py configs/experiments/colour_nfa.yaml --methods biarc clothoid --ends curves
python scripts/make_frescoes.py --n 50 --seed 7 --out datasets/frescoes_50.yaml
```

### Expected results

Pieces in the right place with the current extraction (both fixes in [Extraction](#extraction)). `run_dataset.py`
uses biarc with found ends.

| experiment | pictures | found ends (`run_dataset.py`) | exact ends (`compare.py`) |
|---|---|---|---|
| `canvases_baseline` | 10 canvases, one closed curve each, best single join per side | 74 / 90 | 80 / 90 |
| `multi_lap` | 6 pictures with several curves, one-to-one matching (LAP) | 49 / 54 | 52 / 54 |
| `frescoes_lap` | 20 fresco-like pictures, LAP | 161 / 180 | 177 / 180 |
| `colour_lap` | the same pictures in fresco colours, geometry only | 163 / 180 | 177 / 180 |

Exact ends do not depend on extraction, so their results are the same as before the fixes. The co-circularity columns
of `compare.py`, and the experiments `frescoes_order`, `colour_features`, `colour_clothoid` and `colour_nfa`, have not
been run with the new extraction yet; their results with the original extraction are below.

### Results with the original extraction

Before the two extraction fixes. On these, the package reproduces the original scripts exactly (same pictures, same
ends file for file, same solver totals).

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

`check_pairs.py configs/experiments/colour_features.yaml` gave 364 right, 36 wrong, 114 negative pairs for biarc with
found ends.

## Extraction

`src/curvecont/extraction/ends.py` measures every place where paint touches a piece border:

1. Paint: each pixel's colour distance to its local background (median over `extraction.window` px).
2. Centre line: a ridge detector (the idea of Steger's line detector) finds the middle of each stroke with sub-pixel
   accuracy. Points within a few pixels of the border are not trusted.
3. Fit: the centre line is followed inwards from the border and fitted with a polynomial (up to `extraction.length`
   px), which is extended to the border. Where it crosses gives the position, its slope the direction and its bending
   the curvature (only with at least `extraction.min_arc_for_curvature` px of curve).
4. Width across the stroke and colour along the centre line. Where the width or colour changes along the stroke (two
   strokes meeting), the end is marked `merged` and the score does not use its width and colour.

Short pieces of curve:

- 5 to 15 px of centre line: a straight line is fitted instead of a polynomial, which gives position and direction
  but no curvature. These ends are marked `short_stub`. Before, anything under 15 px was rejected.
- Too small for a centre line (fewer than 8 points): the end is measured from the paint contact on the border. A
  stroke of width w crossing the border at an angle b from straight-on touches it over w / cos b, so the contact
  length gives b, the side where the paint lies gives the lean, and the middle of the contact gives the position.
  These ends get no curvature, their width is not trusted, and they are marked `from_contact`.
- Ends that meet the border more than `extraction.max_incidence_deg` (75) degrees from straight-on are rejected.

Why these two fixes: the gap between found and exact ends came from ends that extraction missed, not from ends
measured imprecisely. Putting the missed ends back from the exact ones closed the whole gap on the frescoes
(148 -> 178), while exact positions and directions alone gave only a few pieces. Most missed ends were short pieces of
curve. Pieces in the right place (biarc, found ends):

| dataset | original | + short stubs | + stubs from the contact | exact ends |
|---|---|---|---|---|
| frescoes | 148 | 159 | 161 | 177 |
| colour | 145 | 160 | 163 | 177 |
| multi | 46 | 49 | 49 | 52 |
| canvases | 71 | 72 | 74 | 80 |
| **total, of 504** | **410** | **440** | **447** | **486** |

Ends found on the frescoes: 1,113, then 1,174, then 1,200 of 1,329.

Tried and not kept: measuring shallow crossings from the paint contact as well (158 on the frescoes instead of 159),
and changing the score of an end without a partner, `score.rules.unmatched_score`, from -0.5 to 0 (157) or -0.25
(161, against 159). The -0.25 gain is small and was tuned on one dataset only.

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
command and the code version (`run.txt`), and its tables (`table.txt`, CSV files). The project is kept in git, so
`run.txt` names the commit of the code that made each result. The solver keeps its solutions in its
`data/experiments/<puzzle>/` folder, in folders named after the score settings, such as `biarc_lap_sol_...` or
`clothoid_exact_lap_w2c1k0.5_nfa_sol_...` (w, c, k = width, colour, curvature weights).

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
tests/                     pytest; tests/data/reference.json holds the numbers the tests check
results/                   written by the scripts (not in git)
thesis/                    text and figures
legacy/                    early stand-alone tools, not used by the pipeline
```

The group's solver is its own repository and is not part of this one.

## Tests

```
pytest                  # everything, about two minutes
pytest -m "not slow"    # without the 64 pair counts on the four test pictures, under a minute
ruff check .
```

`tests/data/reference.json` holds, for four test pictures, the number of exact and found ends and the right / wrong /
negative pairs for eight score settings. The numbers for found ends were recomputed after the two extraction fixes;
the numbers for exact ends are those of the original scripts. When extraction is changed on purpose, the tests on
found ends fail until these numbers are recomputed, which should only be done after checking that the change is an
improvement.

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
- git says `detected dubious ownership`: the folder belongs to the Administrators group, for example because it was
  made from a window opened as administrator. Run `git config --global --add safe.directory <project folder>` in a
  normal window.