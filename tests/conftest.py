"""Shared test data: four reference puzzles made in a temporary folder with the default settings.

tests/data/reference.json holds what the original scripts gave for these four pictures (number of exact and found
ends, and right / wrong / negative pairs for eight score settings); test_regression.py checks the package against it.
"""

import json
from pathlib import Path

import pytest

from curvecont.config import PROJECT, load_config
from curvecont.extraction.ends import extract_pieces
from curvecont.puzzles import load_dataset
from curvecont.synth.curves import curves_from_spec, make_puzzle

REFERENCE = json.loads((Path(__file__).parent / "data" / "reference.json").read_text())


def picture(dataset: str, name: str) -> dict:
    return next(p for p in load_dataset(PROJECT / "datasets" / dataset) if p["name"] == name)


REFERENCE_PICTURES = [
    {"name": "circle", "circle": [300, 300, 190]},  # one curve
    picture("canvases.yaml", "canvas_05"),  # one closed curve, shallow crossings
    picture("multi_curves.yaml", "three"),  # three curves
    picture("frescoes_colour.yaml", "colour_06"),  # 37 coloured bands
]


@pytest.fixture(scope="session")
def reference_data(tmp_path_factory) -> Path:
    """A data folder (as in the solver) with the four reference puzzles and their found ends."""
    data = tmp_path_factory.mktemp("data")
    config = load_config()
    for spec in REFERENCE_PICTURES:
        made = make_puzzle(curves_from_spec(spec, config.picture()), config.picture(), data, spec["name"])
        extract_pieces(sorted((made.folder / "images").glob("piece_*.png")), data / "features" / made.name / "curves", config.extraction())
    return data
