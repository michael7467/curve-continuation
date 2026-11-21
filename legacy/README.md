# legacy

Early stand-alone tools, kept for reference. They are not part of the pipeline, the package does not import them, and
the tests and linter skip this folder. They read the older folder layout (`data/pieces/...`, `results/extract`).

| file | what it did | now |
|---|---|---|
| `biarc_score.py` | biarc score of every pair of pieces in a folder of mixed pieces, written to `questions.csv` | `src/curvecont/scoring/` |
| `cocircular_score.py` | the same with the co-circularity cost | `scoring/geometry.py` (`cocircularity_cost`) |
| `gap_test.py` | how the biarc costs change with the gap between pieces | not needed: the gap comes from the puzzle |
| `export_to_solver.py` | copied pieces and ends into the solver's data folder | `synth/curves.py` writes there directly |

The original pipeline scripts (`make_curve.py`, `extract.py`, `curves.py`, `run_all.py`, `run_canvases.py`,
`compare_scores.py`, `check_scores.py`, ...) were replaced by the package and `scripts/`. Their numbers on four
reference pictures are stored in `tests/data/reference.json`, and `tests/test_regression.py` checks that the package
still gives exactly those numbers.
