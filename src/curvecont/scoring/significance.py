"""A contrario significance of a placement: could its joins happen by chance between unrelated fragments?

Chance model: two ends of unrelated fragments are independent, and each one looks like an end of this puzzle. Random
pairs of whole ends (direction, width, colour and curvature kept together, because in a fresco they go together) are
placed at random positions along two facing edges, and their join cost (features.join_cost, the same cost as the
score) is sampled once. For an observed join, p = the share of random pairs that join at least as well. The model is
built from single ends, never from real pairs, so true joins cannot leak into it. Below the 1% quantile, p follows a
power law fitted on the 0.1% to 1% quantiles, so that small p can still be told apart.

Number of false alarms of a placement whose facing edges have n_a and n_b ends, and whose best joins (one-to-one,
most surprising first) have probabilities p1 <= ... <= pm:

    NFA = tests * C(n_a, m) * C(n_b, m) * m! * p1 * ... * pm, minimised over m,

with tests = N - 1 per edge (the other pieces that could sit there) or N (N - 1) 4 per puzzle (every placement).
A placement is meaningful when NFA < epsilon (Desolneux, Moisan and Morel's a contrario framework).
"""

from __future__ import annotations

from dataclasses import replace
from math import comb, factorial, log10
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .features import join_cost
from .geometry import wrap

if TYPE_CHECKING:
    from ..config import ScoreParams
    from .ends import End

NORMAL = {"R": 0.0, "L": np.pi, "T": -np.pi / 2, "B": np.pi / 2}  # outward direction of each edge (y points down)


class Background:
    """The chance model of one puzzle (see the module text)."""

    def __init__(self, ends: list[list[End]], shapes: list[tuple[int, ...]], step: float, p: ScoreParams):
        pool = [e for piece in ends for e in piece]
        self.tests = max(1, len(ends) - 1) if p.tests == "edge" else max(1, len(ends) * (len(ends) - 1) * 4)
        size = float(np.median([s[0] for s in shapes] + [s[1] for s in shapes]))
        rng = np.random.default_rng(p.seed)
        costs = np.full(p.samples, np.inf)
        if pool:
            shift = np.array([float(step), 0.0])
            for k in range(p.samples):
                ea, eb = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
                a = replace(ea, edge="R", pos=np.array([size, rng.uniform(0, size)]), theta=wrap(ea.theta - NORMAL[ea.edge]))
                b = replace(eb, edge="L", pos=np.array([0.0, rng.uniform(0, size)]), theta=wrap(eb.theta - NORMAL[eb.edge] + np.pi))
                c = join_cost(a, b, shift, p)
                costs[k] = np.inf if c is None else c
        self.costs = np.sort(costs)
        q1, q01 = np.quantile(self.costs, 0.01), np.quantile(self.costs, 0.001)
        self.q1 = float(q1)
        self.power = float(np.clip(np.log(10) / np.log(q1 / q01), 0.3, 2.0)) if np.isfinite(q1) and q01 > 0 and q1 > q01 else 1.0

    def p(self, cost: float | None) -> float:
        """Probability that two unrelated ends join with a cost at most `cost`."""
        if cost is None:
            return 1.0
        if cost >= self.q1:
            return float(np.searchsorted(self.costs, cost, side="right") / len(self.costs))
        return 0.01 * (max(cost, 0.0) / self.q1) ** self.power


def placement_log_nfa(ends_a: list[End], ends_b: list[End], shift: np.ndarray, p: ScoreParams, background: Background) -> float:
    """log10 of the NFA of the best set of joins between two facing edges (see the module text)."""
    logp = np.array([[np.log10(max(background.p(join_cost(a, b, shift, p)), 1e-300)) for b in ends_b] for a in ends_a])
    if p.matching == "best":
        return log10(background.tests) + log10(len(ends_a) * len(ends_b)) + float(logp.min())
    rows, cols = linear_sum_assignment(logp)
    ps = np.sort(logp[rows, cols])
    n_a, n_b = len(ends_a), len(ends_b)
    return min(log10(background.tests) + log10(comb(n_a, m)) + log10(comb(n_b, m)) + log10(factorial(m)) + float(ps[:m].sum())
               for m in range(1, len(ps) + 1))
