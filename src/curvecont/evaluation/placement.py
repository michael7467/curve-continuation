"""Pieces in the right place, from the solver's solution.

Two ways to count, because the solver may shift the whole layout:
    vs_anchor   a piece is right when its grid offset from the anchor piece is the true one (the anchor counts)
    best_shift  the largest group of pieces that are right together, whatever the shift
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..puzzles import is_puzzle_of


def right_vs_anchor(found: np.ndarray, true: np.ndarray, anchor: int) -> int:
    """found, true: (N, 2) grid cells (row, col) of every piece."""
    return int(np.all(found - found[anchor] == true - true[anchor], axis=1).sum())


def right_best_shift(found: np.ndarray, true: np.ndarray) -> int:
    return max(Counter(map(tuple, found - true)).values())


def read_solution(solution: Path, layout: list[dict], step: float) -> tuple[np.ndarray, np.ndarray]:
    """Grid cells found by the solver (from solution.txt: name, x, y of the piece centre) and the true cells."""
    cells = {}
    for line in solution.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            cells[parts[0]] = (round(float(parts[2]) / step), round(float(parts[1]) / step))
    names = [p["name"] for p in layout if p["name"] in cells]
    found = np.array([cells[n] for n in names])
    true = np.array([(p["row"], p["col"]) for p in layout if p["name"] in cells])
    return found, true


def latest_solution(data: Path, picture: str, tag: str) -> tuple[int, int] | None:
    """(right with the best shift, pieces) of the newest solver run of a picture with this settings tag, or None."""
    runs = sorted((p for p in data.glob(f"experiments/{picture}_*/*/{tag}_sol_*/solution.txt") if is_puzzle_of(picture, p.parents[2].name)),
                  key=lambda p: p.parent.name)
    if not runs:
        return None
    pre = data / "preprocessing" / runs[-1].parents[2].name
    step = json.loads((pre / "curves_grid.json").read_text())["xy_step"]
    layout = json.loads((pre / "layout.json").read_text())["pieces"]
    found, true = read_solution(runs[-1], layout, step)
    return right_best_shift(found, true), len(layout)
