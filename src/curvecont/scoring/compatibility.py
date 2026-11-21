"""The compatibility S of one placement (piece j at an offset from piece i), and the matrix R for the solver.

    overlap                          S = -1
    farther apart than max_gap       S = 0
    diagonal (corners on)            best join of ends near the shared corner whose biarc does not cut a piece
    side by side                     the ends on the two facing edges, scored by the matching (matching.py);
                                     both edges empty gives 0
    significance = nfa               a smooth side placement (S > 0) keeps support only when its joins are unlikely by
                                     chance: S = min(1, log10(epsilon / NFA) / 3), or 0 if NFA >= epsilon;
                                     contradictions (S <= 0) keep their score
"""

from __future__ import annotations

from math import log10
from typing import TYPE_CHECKING

import numpy as np

from .features import join_cost
from .geometry import biarc, cuts_a_piece
from .matching import best_join, side_score
from .significance import Background, placement_log_nfa

if TYPE_CHECKING:
    from ..config import ScoreParams
    from .ends import End

FACING = {"right": ("R", "L"), "left": ("L", "R"), "below": ("B", "T"), "above": ("T", "B")}
NFA_DECADES = 3.0  # S reaches 1 when the NFA is 1000 times smaller than epsilon


def corner_score(ends_i, ends_j, wi, hi, wj, hj, ox, oy, p: ScoreParams) -> float:
    """Diagonal placement: the best join between ends near the shared corner that passes through the gap."""
    right, below = ox > 0, oy > 0
    corner_i = np.array([wi if right else 0, hi if below else 0], float)
    corner_j = np.array([0 if right else wj, 0 if below else hj], float)
    edges_i = ("R" if right else "L", "B" if below else "T")
    edges_j = ("L" if right else "R", "T" if below else "B")
    shift = np.array([ox, oy], float)
    near_i = [e for e in ends_i if e.edge in edges_i and np.hypot(*(e.pos - corner_i)) < p.corner_px]
    near_j = [e for e in ends_j if e.edge in edges_j and np.hypot(*(e.pos - corner_j)) < p.corner_px]
    if not near_i or not near_j:
        return 0.0
    cells = [(0, wi, 0, hi), (ox, ox + wj, oy, oy + hj), (ox, ox + wj, 0, hi), (0, wi, oy, oy + hj)]
    costs = []
    for a in near_i:
        for b in near_j:
            bi = biarc(a.pos, a.theta, b.pos + shift, b.theta + np.pi)
            if bi is not None and not cuts_a_piece(bi, cells, (a.width + b.width) / 2):
                costs.append(join_cost(a, b, shift, p))
    return best_join(costs, p)


def placement_score(ends_i: list[End], ends_j: list[End], shape_i, shape_j, dx: float, dy: float, p: ScoreParams,
                    background: Background | None = None) -> float:
    """S for piece j whose centre is (dx, dy) px from the centre of piece i (shapes are (height, width, ...))."""
    hi, wi = shape_i[:2]
    hj, wj = shape_j[:2]
    ox = dx - wj // 2 + wi // 2
    oy = dy - hj // 2 + hi // 2
    apart_x = max(ox - wi, -(ox + wj))
    apart_y = max(oy - hi, -(oy + hj))
    if apart_x < 0 and apart_y < 0:
        return -1.0
    if max(apart_x, apart_y) > p.max_gap:
        return 0.0
    if apart_x >= 0 and apart_y >= 0:
        return corner_score(ends_i, ends_j, wi, hi, wj, hj, ox, oy, p) if p.corners else 0.0
    if apart_x >= 0:
        where = "right" if ox > 0 else "left"
        low, high, axis = max(0, oy), min(hi, oy + hj), 1
    else:
        where = "below" if oy > 0 else "above"
        low, high, axis = max(0, ox), min(wi, ox + wj), 0
    edge_i, edge_j = FACING[where]
    shift = np.array([ox, oy], float)
    ends_a = [e for e in ends_i if e.edge == edge_i and low <= e.pos[axis] <= high]
    ends_b = [e for e in ends_j if e.edge == edge_j and low <= e.pos[axis] + shift[axis] <= high]
    if not ends_a and not ends_b:
        return 0.0
    S = side_score(ends_a, ends_b, shift, axis, low, high, p)
    if p.significance == "off" or S <= 0 or not ends_a or not ends_b:
        return S
    if background is None:
        raise ValueError("significance nfa needs the chance model: pass background=Background(...)")
    log_eps = log10(p.epsilon)
    log_nfa = placement_log_nfa(ends_a, ends_b, shift, p, background)
    return min(1.0, (log_eps - log_nfa) / NFA_DECADES) if log_nfa < log_eps else 0.0


def chance_model(ends: list[list[End]], shapes: list, step: float, p: ScoreParams) -> Background | None:
    """The chance model when significance is on, else None."""
    return Background(ends, shapes, step, p) if p.significance == "nfa" else None


def compatibility_matrix(ends: list[list[End]], shapes: list, offsets: np.ndarray, centre: float, step: float,
                         p: ScoreParams) -> np.ndarray:
    """R[x, y, 0, j, i] = S of piece j at grid point (x, y) around piece i, as the group's solver expects it.
    offsets[y, x] holds the canvas position of each grid point, centre is the canvas centre, step the grid step (px)."""
    background = chance_model(ends, shapes, step, p)
    N, n = len(ends), offsets.shape[0]
    R = np.zeros((n, n, 1, N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                for x in range(n):
                    for y in range(n):
                        xj, yj = offsets[y, x]
                        R[x, y, 0, j, i] = placement_score(ends[i], ends[j], shapes[i], shapes[j], xj - centre, yj - centre, p, background)
    return R
