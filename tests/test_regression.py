"""The package against the numbers of the original scripts: right / wrong / negative pairs on the four reference
puzzles, for eight score settings and both kinds of ends (64 counts). Takes about a minute; skip it with -m "not slow"."""

import pytest

from conftest import REFERENCE
from curvecont.config import ScoreParams
from curvecont.evaluation.pairs import count_pairs

ALL_FEATURES = dict(w_width=2.0, w_colour=1.0, w_curvature=0.5)
SETTINGS = {
    "biarc_best": dict(method="biarc", matching="best"),
    "biarc_lap": dict(method="biarc", matching="lap"),
    "cocirc_lap": dict(method="cocircularity", matching="lap"),
    "biarc_order_skip": dict(method="biarc", matching="order", skip_bad_joins=True),
    "biarc_lap_weights": dict(method="biarc", matching="lap", **ALL_FEATURES),
    "clothoid_lap_weights_cc": dict(method="clothoid", matching="lap", curvature_check="clothoid", **ALL_FEATURES),
    "biarc_lap_weights_nfa_edge": dict(method="biarc", matching="lap", significance="nfa", tests="edge", **ALL_FEATURES),
    "clothoid_lap_nfa_puzzle": dict(method="clothoid", matching="lap", significance="nfa", tests="puzzle"),
}


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(REFERENCE["pairs"]))
def test_pair_counts(reference_data, key):
    puzzle, settings, ends = key.split("|")
    p = ScoreParams(ends=ends, **SETTINGS[settings])
    assert list(count_pairs(reference_data, puzzle, p)) == REFERENCE["pairs"][key]
