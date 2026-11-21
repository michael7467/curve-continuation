"""Drawing, cutting and extraction on the reference pictures, and the counting of pieces in the right place."""

import json

import numpy as np
import pytest

from conftest import REFERENCE
from curvecont.config import load_config
from curvecont.evaluation.ends import compare_ends
from curvecont.evaluation.placement import right_best_shift, right_vs_anchor
from curvecont.solver_bridge import solver_params


def count(folder, ok_only: bool) -> int:
    return sum(sum(1 for e in json.loads(f.read_text())["ends"] if e["ok"] or not ok_only) for f in folder.glob("*_ends.json"))


@pytest.mark.parametrize("puzzle", sorted(REFERENCE["ends"]))
def test_number_of_ends(reference_data, puzzle):
    features = reference_data / "features" / puzzle
    want = REFERENCE["ends"][puzzle]
    assert count(features / "exact", ok_only=False) == want["exact"]
    assert count(features / "curves", ok_only=True) == want["found_ok"]


def test_circle_ends_are_found_accurately(reference_data):
    features = reference_data / "features" / "circle_3x3_00000"
    rows, extra = compare_ends(features / "exact", features / "curves", tolerance=5.0)
    assert (len(rows), sum(r["found"] for r in rows), extra) == (16, 16, 0)
    assert max(r["position_error_px"] for r in rows) < 0.5
    assert max(abs(r["direction_error_deg"]) for r in rows) < 2.0
    assert max(abs(r["curvature_error"]) for r in rows) < 0.003  # the circle's curvature is 1/190 = 0.0053


def test_pieces_in_the_right_place():
    true = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    shifted = true + [1, 0]  # the whole layout one row down: every piece right relative to the others
    assert right_vs_anchor(shifted, true, anchor=0) == 4
    assert right_best_shift(shifted, true) == 4
    swapped = true[[0, 1, 3, 2]]  # two pieces exchanged
    assert right_vs_anchor(swapped, true, anchor=0) == 2
    assert right_best_shift(swapped, true) == 2


def test_solver_params_enable_only_the_curves_feature():
    defaults = {"preprocessing": {}, "compatibility": {"features": {"lines": {"enabled": True}}}, "aggregation": {},
                "solver": {"grid": {}}}
    config = load_config()
    params = solver_params(defaults, config, "circle_3x3_00000", {"piece_size": 185, "xy_step": 200}, config.score())
    features = params["compatibility"]["features"]
    assert not features["lines"]["enabled"]
    assert features["curves"]["enabled"] and features["curves"]["score"] == "biarc" and features["curves"]["matching"] == "lap"
    assert params["compatibility"]["grid"]["xy_step"] == 200 and params["aggregation"]["method"] == "curves"
    assert params["solver"]["no_rotations"] and params["solver"]["anchor_index"] == 0
