"""Right and wrong pairs: does the score agree with the true layout, for every pair of pieces and each of the 8 neighbours?

    python scripts/check_pairs.py configs/experiments/colour_features.yaml
    python scripts/check_pairs.py configs/experiments/colour_nfa.yaml --methods biarc clothoid

For found and for exact ends: right pairs with S > 0, wrong pairs with S > 0 (false support), right pairs with S < 0
(false contradiction). The last columns are the pieces in the right place in the newest solver run with these
settings and found ends (run scripts/compare.py or scripts/run_dataset.py for them). The solver is not run here.
"""

import argparse
import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from curvecont.config import ENDS, METHODS, load_config, new_results_folder, save_run_info
from curvecont.evaluation.pairs import count_pairs
from curvecont.evaluation.placement import latest_solution
from curvecont.puzzles import latest_puzzle, load_dataset

LONG = {"biarc": "biarc", "cocircularity": "co-circularity", "clothoid": "clothoid"}
SHORT = {"biarc": "biarc", "cocircularity": "co-circ", "clothoid": "clothoid"}
ENDS_LABEL = {"curves": "found ends (extraction)", "exact": "exact ends (equations)"}
CELL, SOLVER_CELL = 18, 10  # column widths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="experiment file")
    ap.add_argument("--dataset", help="dataset file instead of the experiment's")
    ap.add_argument("--methods", nargs="+", choices=METHODS, help="score methods (default: compare.methods of the experiment)")
    ap.add_argument("--ends", nargs="+", choices=ENDS, help="kinds of ends (default: compare.ends of the experiment)")
    a = ap.parse_args()
    config = load_config(a.config)
    dataset = Path(a.dataset) if a.dataset else config.dataset
    if dataset is None:
        raise SystemExit("no dataset: give an experiment file with dataset: ..., or --dataset")
    methods, kinds = tuple(a.methods or config.compare_methods), tuple(a.ends or config.compare_ends)
    combos = [(ends, method) for ends in kinds for method in methods]
    out = new_results_folder(config.name, "pairs")
    save_run_info(out, config, sys.argv)
    names = [p["name"] for p in load_dataset(dataset)]
    data = config.data_folder
    w = max(12, max(map(len, names)) + 2)
    group = max(CELL * len(methods), max(map(len, ENDS_LABEL.values())) + 2)  # the columns of one kind of ends

    lines = []

    def show(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    def row(label: str, pair_cells: list[str], solver_cells: list[str]) -> str:
        """A table line: the label, the pair counts of each kind of ends padded to the group width, the solver."""
        groups = [pair_cells[k:k + len(methods)] for k in range(0, len(pair_cells), len(methods))]
        return (f"{label:<{w}}" + "".join(f"{''.join(f'{c:<{CELL}}' for c in g):<{group}}" for g in groups)
                + "".join(f"{c:<{SOLVER_CELL}}" for c in solver_cells)).rstrip()

    def pair_cell(r) -> str:
        return f"{r[0]:>3} {r[1]:>3} {r[2]:>3}"

    def solver_cell(s) -> str:
        return f"{s[0]:>2} / {s[1]}" if s and s[1] else " -"

    show(config.score().summary())
    show(f"{'':{w}}" + "".join(f"{ENDS_LABEL[e]:<{group}}" for e in kinds) + "solver, last run")
    show(row("", [LONG[m] for _ in kinds for m in methods], [SHORT[m] for m in methods]))
    total = {c: [0, 0, 0] for c in combos}
    pieces = {m: [0, 0] for m in methods}
    rows = []
    for name in names:
        puzzle = latest_puzzle(data, name)
        if puzzle is None:
            show(f"{name}: no puzzle yet, run  python scripts/run_dataset.py {a.config}  first")
            continue
        cells = []
        for ends, method in combos:
            r = count_pairs(data, puzzle, config.score(method=method, ends=ends))
            total[(ends, method)] = [t + x for t, x in zip(total[(ends, method)], r, strict=True)]
            cells.append(pair_cell(r))
            rows.append({"picture": name, "puzzle": puzzle, "ends": ends, "method": method, "right": r[0], "wrong": r[1], "negative": r[2]})
        solved = []
        for method in methods:
            s = latest_solution(data, name, config.score(method=method, ends="curves").tag())
            if s:
                pieces[method] = [pieces[method][0] + s[0], pieces[method][1] + s[1]]
            solved.append(solver_cell(s))
        show(row(name, cells, solved))
    show(row("total", [pair_cell(t) for t in total.values()], [solver_cell(p) for p in pieces.values()]))
    show("pairs: right with S > 0, wrong with S > 0, right with S < 0   solver: pieces in the right place")
    with open(out / "pairs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["picture", "puzzle", "ends", "method", "right", "wrong", "negative"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print(f"saved in {out.as_posix()}")


if __name__ == "__main__":
    main()
