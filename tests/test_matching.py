"""The three matchings on small sides: piece i's right edge at x = 100, piece j shifted 115 px to the right."""

import numpy as np
import pytest

from curvecont.config import ScoreParams
from curvecont.scoring.ends import End
from curvecont.scoring.matching import best_score, lap_score, order_score

SHIFT = np.array([115.0, 0.0])
SIDE = dict(shift=SHIFT, axis=1, low=0.0, high=101.0)  # the facing part of the two edges runs over y in [0, 101]
CROSS = np.degrees(np.arctan2(40, 15))  # direction from (100, 30) to (115, 70)


def end_i(y: float, deg: float) -> End:
    """End on the right edge of piece i, leaving towards piece j in direction deg."""
    return End("R", np.array([100.0, y]), np.radians(deg))


def end_j(y: float, arriving_deg: float) -> End:
    """End on the left edge of piece j for a curve arriving in direction arriving_deg (theta points out of piece j)."""
    return End("L", np.array([0.0, y]), np.radians(arriving_deg) - np.pi)


def score(matching, a, b, **settings):
    return {"best": best_score, "lap": lap_score, "order": order_score}[matching](a, b, p=ScoreParams(**settings), **SIDE)


@pytest.mark.parametrize("matching", ["best", "lap", "order"])
def test_two_straight_curves_score_one(matching):
    assert score(matching, [end_i(30, 0), end_i(70, 0)], [end_j(30, 0), end_j(70, 0)]) == pytest.approx(1.0)


def test_lap_counts_an_end_without_partner_best_ignores_it():
    a, b = [end_i(30, 0), end_i(70, 0)], [end_j(30, 0)]
    assert score("best", a, b) == pytest.approx(1.0)
    # one perfect join (counted for both its ends) and one lone end at -0.5: (2 * 1 - 0.5) / 3
    assert score("lap", a, b) == pytest.approx(0.5)


def test_lone_end_near_a_corner_counts_zero():
    a, b = [end_i(30, 0), end_i(90, 0)], [end_j(30, 0)]  # y = 90 is 11 px from the corner at 101
    assert score("lap", a, b) == pytest.approx(2 / 3)
    assert score("lap", a, b, corners=False) == pytest.approx(0.5)


def test_order_refuses_crossing_joins():
    # the two curves cross in the gap: lap may pair them that way, order may not
    a, b = [end_i(30, CROSS), end_i(70, -CROSS)], [end_j(30, -CROSS), end_j(70, CROSS)]
    assert score("lap", a, b) == pytest.approx(1.0)
    assert score("order", a, b) == pytest.approx(-1.0)  # forced into the two non-crossing joins
    # with skip_bad_joins it keeps one crossing join and leaves the other two ends alone: (2 * 1 - 0.5 - 0.5) / 4
    assert score("order", a, b, skip_bad_joins=True) == pytest.approx(0.25)


@pytest.mark.parametrize("matching", ["lap", "order"])
def test_skip_bad_joins(matching):
    a, b = [end_i(50, 0)], [end_j(50, -90)]  # the curve would have to turn back up: a bad join
    assert score(matching, a, b) == pytest.approx(-1.0)
    assert score(matching, a, b, skip_bad_joins=True) == pytest.approx(-0.5)


def test_empty_side_gets_the_unmatched_score():
    assert score("best", [end_i(50, 0)], []) == pytest.approx(-0.5)
    assert score("lap", [end_i(50, 0)], []) == pytest.approx(-0.5)
    assert score("order", [], [end_j(50, 0)]) == pytest.approx(-0.5)
