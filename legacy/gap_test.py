import argparse
from pathlib import Path

import numpy as np

from biarc_score import candidate_costs, read_folder, run_tests


def cost_table(pieces, place, gap):
    names = list(pieces)
    placements, questions = run_tests(pieces, place, gap, False)
    table, truth = [], []
    for q in questions:
        cost = candidate_costs(placements, names, q["piece"], q["true_where"], False)
        table.append(list(cost.values()))
        truth.append(list(cost).index((q["true_neighbour"], q["true_where"])))
    return np.array(table), np.array(truth)


def right(table, truth):
    rows = np.arange(len(truth))
    true_cost = table[rows, truth]
    wrong = table.copy()
    wrong[rows, truth] = np.inf
    return int(np.sum(np.isfinite(true_cost) & (true_cost < wrong.min(axis=1))))


def clear_winners(table):
    best = np.sort(table, axis=1)[:, :2]
    return int(np.sum(np.isfinite(best[:, 1]) & (best[:, 0] < 0.5 * best[:, 1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="a folder of mixed pieces such as data/pieces/all_3x3")
    ap.add_argument("--gaps", type=float, nargs=2, default=[0, 32], metavar=("FROM", "TO"), help="range of gaps to try in px (default 0 32)")
    ap.add_argument("--step", type=float, default=2, help="step between the gaps in px (default 2)")
    ap.add_argument("--ends", default="results/extract", help="where extract.py wrote its files (default results/extract)")
    a = ap.parse_args()
    folder = Path(a.folder)
    layout, place, pieces = read_folder(folder, a.ends)
    gaps = np.arange(a.gaps[0], a.gaps[1] + 1e-9, a.step)
    results = [cost_table(pieces, place, g) for g in gaps]
    tables, truth = [t for t, _ in results], results[0][1]
    n = len(truth)
    clear = [clear_winners(t) for t in tables]
    top = [i for i, c in enumerate(clear) if c == max(clear)]
    pick = top[len(top) // 2]
    true_gap = 2 * layout["margin"]
    print(f"{folder.as_posix()}: {len(pieces)} pieces, {n} questions, gaps tried {a.gaps[0]:g} to {a.gaps[1]:g} px (step {a.step:g})")
    print(f"  best gap for every pair: true neighbour first in {right(np.min(tables, axis=0), truth)} / {n}")
    print(f"  one gap for all pieces, estimated from the data: {gaps[pick]:g} px "
          f"(most clear winners, {max(clear)}, at {', '.join(f'{gaps[i]:g}' for i in top)} px): true neighbour first in {right(tables[pick], truth)} / {n}")
    print(f"  true gap {true_gap:g} px, for comparison: true neighbour first in {right(*cost_table(pieces, place, true_gap))} / {n}")


if __name__ == "__main__":
    main()
