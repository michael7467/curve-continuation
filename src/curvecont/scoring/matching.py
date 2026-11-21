"""From the join scores of the ends on two facing edges to one score for the placement.

Join score of two ends: s = 1 - cost / tau, clipped to [-1, 1]. An end with no partner gets unmatched_score, or 0 when
it is near a corner (the curve may continue into the diagonal piece).

best   S = the best single join; other ends are ignored (an empty side gives the lone value)
lap    the ends are paired one to one (linear assignment, highest total); every end gets the score of its join or
       its lone value, and S is the mean over all ends of both edges
order  like lap, but the pairs keep their order along the edge (no crossing in the gap), by dynamic programming as
       in sequence alignment
skip_bad_joins (lap, order): a join may be refused when leaving both ends alone scores better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import join_cost

if TYPE_CHECKING:
    from ..config import ScoreParams
    from .ends import End


def join_scores(ends_a: list[End], ends_b: list[End], shift: np.ndarray, p: ScoreParams) -> np.ndarray:
    """Matrix of join scores (NaN where no join is possible)."""
    s = np.full((len(ends_a), len(ends_b)), np.nan)
    for i, a in enumerate(ends_a):
        for j, b in enumerate(ends_b):
            c = join_cost(a, b, shift, p)
            if c is not None:
                s[i, j] = np.clip(1.0 - c / p.tau, -1.0, 1.0)
    return s


def lone_value(along: float, low: float, high: float, p: ScoreParams) -> float:
    """Value of an end left without a partner, at position `along` on an edge running from low to high."""
    if p.corners and min(along - low, high - along) < p.corner_px:
        return 0.0
    return p.unmatched_score


def best_score(ends_a, ends_b, shift, axis, low, high, p: ScoreParams) -> float:
    if not ends_a or not ends_b:
        lone = [e.pos[axis] for e in ends_a] + [e.pos[axis] + shift[axis] for e in ends_b]
        if p.corners and all(min(v - low, high - v) < p.corner_px for v in lone):
            return 0.0
        return p.unmatched_score
    return best_join([join_cost(a, b, shift, p) for a in ends_a for b in ends_b], p)


def best_join(costs: list[float | None], p: ScoreParams) -> float:
    """Score of the cheapest possible join; the lone value when no join is possible."""
    costs = [c for c in costs if c is not None]
    if not costs:
        return p.unmatched_score
    return float(np.clip(1.0 - min(costs) / p.tau, -1.0, 1.0))


def lap_score(ends_a, ends_b, shift, axis, low, high, p: ScoreParams) -> float:
    s = join_scores(ends_a, ends_b, shift, p)
    ua = [lone_value(a.pos[axis], low, high, p) for a in ends_a]
    ub = [lone_value(b.pos[axis] + shift[axis], low, high, p) for b in ends_b]
    n, m = len(ends_a), len(ends_b)
    if p.skip_bad_joins:
        # extra rows and columns let every end stay alone at its lone value
        V = np.full((n + m, m + n), -1e6)
        V[:n, :m] = np.where(np.isnan(s), -1e6, 2 * s)
        V[np.arange(n), m + np.arange(n)] = ua
        V[n + np.arange(m), np.arange(m)] = ub
        V[n:, m:] = 0.0
        rows, cols = linear_sum_assignment(V, maximize=True)
        return float(V[rows, cols].sum() / (n + m))
    pairs = []
    if ends_a and ends_b:
        rows, cols = linear_sum_assignment(np.where(np.isnan(s), 10.0, -s))
        pairs = [(i, j) for i, j in zip(rows, cols, strict=True) if not np.isnan(s[i, j])]
    used_a, used_b = {i for i, _ in pairs}, {j for _, j in pairs}
    total = 2 * sum(s[i, j] for i, j in pairs)
    total += sum(ua[i] for i in range(n) if i not in used_a) + sum(ub[j] for j in range(m) if j not in used_b)
    return float(total / (n + m))


def order_score(ends_a, ends_b, shift, axis, low, high, p: ScoreParams) -> float:
    ends_a = sorted(ends_a, key=lambda e: e.pos[axis])
    ends_b = sorted(ends_b, key=lambda e: e.pos[axis] + shift[axis])
    s = join_scores(ends_a, ends_b, shift, p)
    ua = [lone_value(a.pos[axis], low, high, p) for a in ends_a]
    ub = [lone_value(b.pos[axis] + shift[axis], low, high, p) for b in ends_b]
    # V[i][j] = (number of joins, total) for the first i ends of a and the first j ends of b; without skip_bad_joins
    # the most joins win (like lap), with it only the total counts
    rank = (lambda o: o[1]) if p.skip_bad_joins else (lambda o: o)
    V = [[(0, 0.0)] * (len(ends_b) + 1) for _ in range(len(ends_a) + 1)]
    for i in range(len(ends_a) + 1):
        for j in range(len(ends_b) + 1):
            options = []
            if i:
                options.append((V[i - 1][j][0], V[i - 1][j][1] + ua[i - 1]))
            if j:
                options.append((V[i][j - 1][0], V[i][j - 1][1] + ub[j - 1]))
            if i and j and not np.isnan(s[i - 1, j - 1]):
                options.append((V[i - 1][j - 1][0] + 1, V[i - 1][j - 1][1] + 2 * s[i - 1, j - 1]))
            if options:
                V[i][j] = max(options, key=rank)
    return float(V[-1][-1][1] / (len(ends_a) + len(ends_b)))


SIDE_SCORES = {"best": best_score, "lap": lap_score, "order": order_score}


def side_score(ends_a, ends_b, shift, axis, low, high, p: ScoreParams) -> float:
    """Score of two facing edges with the configured matching."""
    return SIDE_SCORES[p.matching](ends_a, ends_b, shift, axis, low, high, p)
