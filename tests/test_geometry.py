"""Join costs on curves whose answer is known: straight lines, circles, reversed joins."""

import numpy as np
import pytest

from curvecont.scoring.geometry import biarc_cost, clothoid, clothoid_cost, cocircularity_cost, wrap

COSTS = [biarc_cost, cocircularity_cost, clothoid_cost]


def on_circle(r: float, angle: float) -> tuple[np.ndarray, float]:
    """Point of a circle around the origin and the direction of travel (anticlockwise)."""
    return r * np.array([np.cos(angle), np.sin(angle)]), angle + np.pi / 2


@pytest.mark.parametrize("cost", COSTS)
def test_straight_continuation_costs_nothing(cost):
    assert cost(np.array([0.0, 0.0]), 0.3, 40 * np.array([np.cos(0.3), np.sin(0.3)]), 0.3) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("r, a0, a1", [(100, 0.3, 0.8), (190, -1.0, 0.2), (60, 2.0, 2.4)])
def test_circle(r, a0, a1):
    P0, th0 = on_circle(r, a0)
    P1, th1 = on_circle(r, a1)
    turn = a1 - a0
    assert cocircularity_cost(P0, th0, P1, th1) == pytest.approx(0, abs=1e-12)
    # biarc and clothoid cost = length x integral of curvature^2 = turning angle^2 on a circle
    assert biarc_cost(P0, th0, P1, th1) == pytest.approx(turn ** 2, rel=1e-9)
    c = clothoid(P0, th0, P1, th1)
    assert c["cost"] == pytest.approx(turn ** 2, rel=1e-9)
    assert c["k0"] == pytest.approx(1 / r, rel=1e-6)
    assert c["k1"] == pytest.approx(1 / r, rel=1e-6)
    assert c["L"] == pytest.approx(r * turn, rel=1e-6)


@pytest.mark.parametrize("cost", [biarc_cost, clothoid_cost])
def test_same_cost_in_both_directions(cost):
    rng = np.random.default_rng(3)
    for _ in range(50):
        P0, P1 = rng.uniform(-50, 50, 2), rng.uniform(-50, 50, 2)
        th0, th1 = rng.uniform(-np.pi, np.pi, 2)
        forward = cost(P0, th0, P1, th1)
        backward = cost(P1, wrap(th1 + np.pi), P0, wrap(th0 + np.pi))
        if forward is not None and backward is not None:
            assert forward == pytest.approx(backward, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("cost", COSTS)
def test_bending_costs_more(cost):
    P0, P1 = np.array([0.0, 0.0]), np.array([30.0, 0.0])
    costs = [cost(P0, 0.0, P1, turn) for turn in (0.0, 0.2, 0.5)]
    assert costs[0] < costs[1] < costs[2]


@pytest.mark.parametrize("cost", COSTS)
def test_same_point_has_no_join(cost):
    assert cost(np.array([5.0, 5.0]), 0.0, np.array([5.0, 5.0]), 0.0) is None
