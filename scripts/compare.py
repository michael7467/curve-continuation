"""Pieces in the right place for several scores, with found and with exact ends, on the newest puzzle of every picture.

    python scripts/compare.py configs/experiments/frescoes_lap.yaml
    python scripts/compare.py configs/experiments/colour_nfa.yaml --methods biarc clothoid

All columns use the same settings; only the score method and the kind of ends change. Make the puzzles first with
scripts/run_dataset.py (any experiment on the same dataset).
"""

import argparse
import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from curvecont.config import ENDS, METHODS, load_config, new_results_folder, save_run_info, save_settings
from curvecont.puzzles import latest_puzzle, load_dataset
from curvecont.solver_bridge import run_solver

SHORT = {"biarc": "biarc", "cocircularity": "co-circ", "clothoid": "clothoid"}
ENDS_LABEL = {"curves": "found ends (extraction)", "exact": "exact ends (equations)"}
CELL = 12  # width of one column


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
    combos = [(method, ends) for ends in kinds for method in methods]
    out = new_results_folder(config.name, "compare")
    save_run_info(out, config, sys.argv)
    settings = {(m, e): save_settings(config.with_score(method=m, ends=e), out / "settings" / f"{m}_{e}.yaml") for m, e in combos}
    names = [p["name"] for p in load_dataset(dataset)]
    w = max(12, max(map(len, names)) + 2)
    group = max(CELL * len(methods), max(map(len, ENDS_LABEL.values())) + 2)  # the columns of one kind of ends

    lines = []

    def show(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    def row(label: str, cells: list[str]) -> str:
        """A table line: the label, then the cells of each kind of ends padded to the group width."""
        groups = [cells[k:k + len(methods)] for k in range(0, len(cells), len(methods))]
        return f"{label:<{w}}" + "".join(f"{''.join(f'{c:<{CELL}}' for c in g):<{group}}" for g in groups).rstrip()

    show(config.score().summary())
    show(f"{'':{w}}" + "".join(f"{ENDS_LABEL[e]:<{group}}" for e in kinds).rstrip())
    show(row("", [SHORT[m] for _ in kinds for m in methods]))
    totals = {c: [0, 0] for c in combos}
    rows = []
    for name in names:
        puzzle = latest_puzzle(config.data_folder, name)
        if puzzle is None:
            show(f"{name}: no puzzle yet, run  python scripts/run_dataset.py {a.config}  first")
            continue
        cells = []
        for combo in combos:
            try:
                r = run_solver(puzzle, settings[combo])
            except RuntimeError as e:
                cells.append("failed")
                with open(out / "errors.txt", "a") as fh:
                    fh.write(f"{puzzle} {combo}\n{e}\n\n")
                continue
            totals[combo][0] += r.right
            totals[combo][1] += r.pieces
            cells.append(f"{r.right} / {r.pieces}")
            rows.append({"picture": name, "puzzle": puzzle, "method": combo[0], "ends": combo[1], "tag": r.tag, "right": r.right,
                         "pieces": r.pieces, "solution": r.solution})
        show(row(name, cells))
    show(row("total", [f"{t[0]} / {t[1]}" for t in totals.values()]))
    show("pieces in the right place; the same settings for all, only the score and the ends change")
    with open(out / "solver.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["picture", "puzzle", "method", "ends", "tag", "right", "pieces", "solution"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print(f"saved in {out.as_posix()}")


if __name__ == "__main__":
    main()
