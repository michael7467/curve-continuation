"""Right and wrong pairs: how the scores of all placements compare with the true layout.

For every ordered pair of pieces and every one of the 8 neighbouring positions:
    right  the true placement has S > 0      (support where it should be)
    wrong  a false placement has S > 0       (false positives)
    negative  the true placement has S < 0   (contradiction where there should be support)
"""

from __future__ import annotations

from pathlib import Path

from ..config import ScoreParams
from ..puzzles import load_puzzle
from ..scoring.compatibility import chance_model, placement_score


def count_pairs(data: Path, puzzle: str, p: ScoreParams) -> tuple[int, int, int]:
    """(right, wrong, negative) for one puzzle, with the ends given by p.ends (curves or exact)."""
    pz = load_puzzle(data, puzzle)
    ends = pz.ends(data, p.ends)
    background = chance_model(ends, pz.shapes, pz.step, p)
    right = wrong = negative = 0
    for i, a in enumerate(pz.layout):
        for j, b in enumerate(pz.layout):
            if i == j:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        s = placement_score(ends[i], ends[j], pz.shapes[i], pz.shapes[j], dc * pz.step, dr * pz.step, p, background)
                        true = (b["row"] - a["row"], b["col"] - a["col"]) == (dr, dc)
                        right += true and s > 0
                        wrong += not true and s > 0
                        negative += true and s < 0
    return right, wrong, negative
