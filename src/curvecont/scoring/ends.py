"""Curve ends as the score uses them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import lab


@dataclass(frozen=True)
class End:
    """One curve end on a piece border.

    edge       L, R, T or B
    pos        (x, y) in px, in the piece's own frame (y points down)
    theta      direction of the curve pointing out of the piece, rad
    width      stroke width, px; width_ok is False when extraction marked the end as merged
    curvature  1/px, measured looking outwards; None when missing or not trusted
    lab        CIE Lab colour; None when missing or not trusted
    """

    edge: str
    pos: np.ndarray
    theta: float
    width: float = 6.0
    width_ok: bool = True
    curvature: float | None = None
    lab: np.ndarray | None = None


def load_ends(folder: str | Path, piece: str) -> list[End]:
    """The fitted ends (ok) of one piece from <folder>/<piece>_ends.json (written by extraction or synth)."""
    path = Path(folder) / f"{piece}_ends.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: extract the curve ends of this puzzle first")
    ends = []
    for e in json.loads(path.read_text())["ends"]:
        if not e["ok"]:
            continue
        ends.append(End(edge=e["edge"], pos=np.array([e["x"], e["y"]], float), theta=float(np.radians(e["theta_deg"])),
                        width=float(e.get("width", 6.0)), width_ok=e.get("width_reliable", True),
                        curvature=e.get("curvature") if e.get("curvature_reliable", True) else None,
                        lab=lab(e["colour"]) if e.get("colour") and e.get("colour_reliable", True) else None))
    return ends
