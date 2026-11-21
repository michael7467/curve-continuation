"""Reading generated puzzles and dataset files.

A puzzle lives in the solver's data folder: preprocessing/<puzzle>/ (pieces, layout.json with the true rows and
columns, curves_grid.json with the grid step) and features/<puzzle>/exact and /curves (the exact and the found ends).
Puzzle names are <picture>_<n>x<n>_<number>; the latest number of a picture is the one scripts use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from .scoring.ends import End, load_ends


@dataclass
class PuzzleData:
    name: str
    folder: Path
    step: float  # px between neighbouring piece centres
    layout: list[dict]  # name, row, col, box of every piece
    shapes: list[tuple[int, int]]  # (height, width) of every piece

    def ends(self, data: Path, kind: str) -> list[list[End]]:
        """The ends of every piece: kind = curves (found by extraction) or exact (from the equations)."""
        return [load_ends(data / "features" / self.name / kind, p["name"]) for p in self.layout]


def load_puzzle(data: Path, puzzle: str) -> PuzzleData:
    folder = data / "preprocessing" / puzzle
    step = json.loads((folder / "curves_grid.json").read_text())["xy_step"]
    layout = json.loads((folder / "layout.json").read_text())["pieces"]
    shapes = [Image.open(folder / "images" / f"{p['name']}.png").size[::-1] for p in layout]
    return PuzzleData(puzzle, folder, step, layout, shapes)


def is_puzzle_of(picture: str, folder_name: str) -> bool:
    return re.fullmatch(rf"{re.escape(picture)}_\d+x\d+_\d+", folder_name) is not None


def latest_puzzle(data: Path, picture: str) -> str | None:
    """The newest puzzle made from this picture that has both exact and found ends, or None."""
    found = sorted(p.name for p in (data / "preprocessing").glob(f"{picture}_*") if is_puzzle_of(picture, p.name)
                   and all((data / "features" / p.name / k).exists() for k in ("exact", "curves")))
    return found[-1] if found else None


def load_dataset(path: Path) -> list[dict]:
    """The picture specs of a dataset file (key pictures:, or canvases: in older files)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    pictures = data.get("pictures", data.get("canvases"))
    if not pictures:
        raise ValueError(f"{path}: no pictures (expected a list under pictures:)")
    for p in pictures:
        if "name" not in p:
            raise ValueError(f"{path}: every picture needs a name")
    return pictures
