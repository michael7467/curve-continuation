import argparse
import glob
from pathlib import Path

import numpy as np

import biarc_score


def cocircular(P0, th0, P1, th1):
    v = np.asarray(P1, float) - np.asarray(P0, float)
    if v @ v < 1e-12:
        return None
    chord = np.arctan2(v[1], v[0])
    error = biarc_score.wrap(th0 - chord) + biarc_score.wrap(th1 - chord)
    return {"cost": float(error ** 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="a folder of mixed pieces such as data/pieces/all_3x3, or a pattern")
    ap.add_argument("--ends", default="results/extract", help="where extract.py wrote its files (default results/extract)")
    ap.add_argument("--out", default="results/cocircular", help="output folder for questions.csv (default results/cocircular)")
    ap.add_argument("--tau", type=float, default=1.0, help="cost in rad^2 at which the score reaches 0 (default 1.0)")
    ap.add_argument("--gap", type=float, help="gap between neighbouring pieces in px (default: 2 x margin from layout.json)")
    a = ap.parse_args()
    a.diagonal = False
    biarc_score.biarc_closed = cocircular
    folders = []
    for item in a.folders:
        folders += [Path(m) for m in sorted(glob.glob(item)) or [item] if Path(m).is_dir()]
    if not folders:
        raise SystemExit("No folder found. Give a folder of pieces such as data/pieces/all_3x3, or a pattern.")
    print("cost: cocircularity error squared")
    for folder in folders:
        biarc_score.process(folder, a)


if __name__ == "__main__":
    main()
