"""Every picture of a dataset through the whole pipeline, with the score settings of the experiment.

    python scripts/run_dataset.py configs/experiments/frescoes_lap.yaml
    python scripts/run_dataset.py configs/experiments/frescoes_lap.yaml --dataset datasets/multi_curves.yaml

Prints, per picture, how many exact curve ends extraction found and how many pieces the solver put in the right
place. This also makes the puzzles that scripts/compare.py and scripts/check_pairs.py use.
"""

import argparse
import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from curvecont.config import load_config, new_results_folder, save_run_info
from curvecont.pipeline import run_picture
from curvecont.puzzles import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="experiment file (its dataset is run), or a settings file together with --dataset")
    ap.add_argument("--dataset", help="dataset file to run instead of the experiment's")
    ap.add_argument("--no-solver", action="store_true", help="only make the pictures, find the ends and compare them")
    a = ap.parse_args()
    config = load_config(a.config)
    dataset = Path(a.dataset) if a.dataset else config.dataset
    if dataset is None:
        raise SystemExit("no dataset: give an experiment file with dataset: ..., or --dataset")
    pictures = load_dataset(dataset)
    out = new_results_folder(config.name, "dataset")
    settings = save_run_info(out, config, sys.argv)
    print(f"{dataset.name}: {len(pictures)} pictures, score {config.score().tag()} ({config.score().summary()})")
    rows, right, total = [], 0, 0
    for spec in pictures:
        try:
            r = run_picture(spec, config, settings, out, solve=not a.no_solver)
        except (RuntimeError, ValueError) as e:
            print(f"{spec['name']}: failed\n{e}", flush=True)
            continue
        line = f"{r.name}: curve ends found {r.found_ends} / {r.exact_ends}"
        row = {"picture": r.name, "puzzle": r.puzzle, "exact_ends": r.exact_ends, "found_ends": r.found_ends}
        if r.solver:
            right, total = right + r.solver.right, total + r.solver.pieces
            line += f", pieces in the right place {r.solver.right} / {r.solver.pieces}"
            row.update(tag=r.solver.tag, right=r.solver.right, pieces=r.solver.pieces, solution=r.solver.solution)
        rows.append(row)
        print(line, flush=True)
    if not a.no_solver:
        print(f"total: {right} / {total} pieces in the right place")
    with open(out / "pictures.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["picture", "puzzle", "exact_ends", "found_ends", "tag", "right", "pieces", "solution"],
                                restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved in {out.as_posix()}")


if __name__ == "__main__":
    main()
