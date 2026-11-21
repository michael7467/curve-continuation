"""The extra join costs (width, colour, curvature) and the reliability flags that switch them off."""

import numpy as np
import pytest

from curvecont.config import ScoreParams
from curvecont.scoring.ends import End
from curvecont.scoring.features import join_cost, lab

SHIFT = np.array([115.0, 0.0])


def pair(**changes_b):
    """Two ends of one straight horizontal curve (geometry cost 0); changes_b alters the second end."""
    a = End("R", np.array([100.0, 50.0]), 0.0, width=6.0, curvature=0.01, lab=lab([160, 62, 45]))
    b = End("L", np.array([0.0, 50.0]), np.pi, width=6.0, curvature=-0.01, lab=lab([160, 62, 45]))
    return a, End(**{**b.__dict__, **changes_b})


def test_lab():
    # white and black up to the rounding of the sRGB matrix; a Lab difference of 2 is barely visible
    assert lab([255, 255, 255]) == pytest.approx([100, 0, 0], abs=0.05)
    assert lab([0, 0, 0]) == pytest.approx([0, 0, 0], abs=0.05)
    assert np.linalg.norm(lab([160, 62, 45]) - lab([30, 60, 160])) > 50  # fresco red against the default blue


def test_features_off_by_default():
    a, b = pair(width=9.0, lab=lab([30, 60, 160]), curvature=0.02)
    assert join_cost(a, b, SHIFT, ScoreParams()) == pytest.approx(0, abs=1e-12)


def test_width():
    a, b = pair(width=8.0)
    assert join_cost(a, b, SHIFT, ScoreParams(w_width=2)) == pytest.approx(2 * (2 * 2 / 14) ** 2)
    a, b = pair(width=8.0, width_ok=False)  # merged end: the width is not trusted
    assert join_cost(a, b, SHIFT, ScoreParams(w_width=2)) == pytest.approx(0, abs=1e-12)


def test_colour():
    a, b = pair(lab=lab([160, 62, 45]) + np.array([0.0, 25.0, 0.0]))
    assert join_cost(a, b, SHIFT, ScoreParams(w_colour=1)) == pytest.approx((25 / 50) ** 2)
    a, b = pair(lab=None)
    assert join_cost(a, b, SHIFT, ScoreParams(w_colour=1)) == pytest.approx(0, abs=1e-12)


def test_curvature_compares_the_two_ends():
    # each end measures its bending looking outwards, so a curve that goes on bending has opposite signs
    a, b = pair(curvature=-0.01)
    assert join_cost(a, b, SHIFT, ScoreParams(w_curvature=0.5)) == pytest.approx(0, abs=1e-12)
    a, b = pair(curvature=0.01)
    assert join_cost(a, b, SHIFT, ScoreParams(w_curvature=0.5)) == pytest.approx(0.5 * (0.02 * 100) ** 2)


def test_curvature_against_the_clothoid():
    # the joining clothoid is straight here (k0 = k1 = 0), so each end pays for its own curvature
    a, b = pair(curvature=-0.01)
    p = ScoreParams(w_curvature=0.5, curvature_check="clothoid")
    assert join_cost(a, b, SHIFT, p) == pytest.approx(0.5 * ((0.01 * 100) ** 2 + (0.01 * 100) ** 2))
