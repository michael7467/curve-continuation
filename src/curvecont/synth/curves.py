"""Draw curves given by their equations, cut the picture into pieces, and compute the exact curve ends.

A curve is x(t), y(t) for t in [t0, t1], written as text such as "300 + 190*cos(t)". A picture spec (from a dataset
file or the curve section of the settings) is one of:

    circle: [cx, cy, r]
    r: "<radius as a function of t>"        a closed curve around the picture centre, t from 0 to 6.283185
    x: "...", y: "...", t: [t0, t1]
    many: [ {circle: ...} or {x, y, t}, ... ]   each item may have its own thickness and colour

plus an optional thickness (px) for the whole picture. Pieces are written where the solver reads them:
<data>/preprocessing/<puzzle>/ (images, binary_masks, layout.json, ground_truth.json, curves_grid.json,
equation.json, full_image.png) and the exact ends to <data>/features/<puzzle>/exact/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from ..config import PictureParams

FUNCTIONS = {"pi": np.pi, "cos": np.cos, "sin": np.sin, "tan": np.tan, "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs}
DEFAULT_COLOUR = (30, 60, 160)
OUTWARD = {"L": (-1.0, 0.0), "R": (1.0, 0.0), "T": (0.0, -1.0), "B": (0.0, 1.0)}
POLAR_TURN = 6.283185  # t range of polar curves given by r (as in the original scripts)


@dataclass
class Curve:
    """One painted curve: x(t), y(t) as text, its t range, thickness (px) and colour (RGB)."""

    x: str
    y: str
    t: list[float]
    thickness: float
    colour: tuple[int, int, int] = DEFAULT_COLOUR
    ts: np.ndarray = field(default=None, repr=False)
    points: np.ndarray = field(default=None, repr=False)

    def at(self, t) -> np.ndarray:
        """Points (n, 2) of the curve at the parameters t."""
        t = np.asarray(t, float)
        names = {**FUNCTIONS, "t": t}
        return np.stack([np.broadcast_to(np.asarray(eval(e, {"__builtins__": {}}, names), float), t.shape)
                         for e in (self.x, self.y)], axis=-1)

    def sample(self) -> None:
        """Dense points along the curve: at least 6000, and about 5 per px of length."""
        t = np.linspace(self.t[0], self.t[1], 6000)
        length = np.hypot(*np.diff(self.at(t), axis=0).T).sum()
        self.ts = np.linspace(self.t[0], self.t[1], int(max(6000, 5 * length)))
        self.points = self.at(self.ts)


@dataclass
class Puzzle:
    name: str
    folder: Path
    pieces: int
    piece_size: tuple[int, int]
    gap: int
    exact_ends: int
    description: str

    def report(self) -> str:
        w, h = self.piece_size
        return (f"{self.folder.as_posix()}: {self.pieces} pieces of {w} x {h} px, gap {self.gap} px, "
                f"{self.exact_ends} exact curve ends   {self.description}")


def curves_from_spec(spec: dict, picture: PictureParams) -> list[Curve]:
    """The curves of a picture spec (see the module text). An older spec may name its form with type: circle,
    equation or many; otherwise the first of many, circle, r, x/y that the spec has is used."""
    thickness = float(spec.get("thickness", picture.thickness))
    kind = spec.get("type") or next((k for k in ("many", "circle", "r") if k in spec), "equation")
    if kind == "many":
        items = spec["many"]
    elif kind == "circle":
        items = [{"circle": spec["circle"]}]
    elif kind == "r":
        centre = picture.size / 2
        items = [{"x": f"{centre:g} + ({spec['r']})*cos(t)", "y": f"{centre:g} + ({spec['r']})*sin(t)", "t": [0, POLAR_TURN]}]
    elif kind == "equation":
        items = [{"x": spec.get("x"), "y": spec.get("y"), "t": spec.get("t")}]
    else:
        raise ValueError(f"curve type must be circle, equation or many, not {kind}")
    curves = []
    for item in items:
        if item.get("circle"):
            cx, cy, r = item["circle"]
            x, y, t = f"{cx:g} + {r:g}*cos(t)", f"{cy:g} + {r:g}*sin(t)", [0.0, 2 * np.pi]
        else:
            x, y, t = item.get("x"), item.get("y"), list(item.get("t") or [0.0, 2 * np.pi])
        if not (x and y):
            raise ValueError('a curve needs circle: [cx, cy, r], r: "...", or both x and y, e.g. x: "300 + 200*cos(t)"')
        colour = item.get("colour", item.get("color", DEFAULT_COLOUR))
        if len(colour) != 3 or not all(0 <= v <= 255 for v in colour):
            raise ValueError(f"colour must be three numbers from 0 to 255, like [160, 62, 45], not {colour}")
        curves.append(Curve(x, y, [float(v) for v in t], float(item.get("thickness", thickness)), tuple(int(v) for v in colour)))
    return curves


def draw(curves: list[Curve], width: int, height: int) -> Image.Image:
    """Anti-aliased drawing; every pixel takes the colour of the curve that covers it most (the first one on a tie)."""
    yy, xx = np.mgrid[0:height, 0:width] + 0.5
    cover = np.zeros(height * width)
    colour = np.zeros((height * width, 3))
    for c in curves:
        dist, _ = cKDTree(c.points).query(np.c_[xx.ravel(), yy.ravel()], distance_upper_bound=c.thickness / 2 + 1)
        this = np.clip(c.thickness / 2 + 0.5 - dist, 0, 1)
        colour[this > cover] = c.colour
        cover = np.maximum(cover, this)
    return Image.fromarray(np.round(255 + cover[:, None] * (colour - 255)).reshape(height, width, 3).astype(np.uint8))


def exact_ends(curve: Curve, box: tuple[int, int, int, int], index: int) -> list[dict]:
    """Where the curve crosses the border of the piece box (x0, y0, x1, y1): position in the piece, outward direction,
    incidence and curvature (1/px, measured looking outwards), refined with Newton steps on the equation."""
    x0, y0, x1, y1 = box
    h = 1e-4 * (curve.t[1] - curve.t[0])
    ends = []
    for edge, axis, value, low, high in (("L", 0, x0, y0, y1), ("R", 0, x1, y0, y1), ("T", 1, y0, x0, x1), ("B", 1, y1, x0, x1)):
        f = curve.points[:, axis] - value
        for i in np.flatnonzero(f[:-1] * f[1:] < 0):
            tc = curve.ts[i] + (curve.ts[i + 1] - curve.ts[i]) * f[i] / (f[i] - f[i + 1])
            for _ in range(3):
                before, p, after = curve.at([tc - h, tc, tc + h])
                d1, d2 = (after - before) / (2 * h), (after - 2 * p + before) / h ** 2
                if abs(d1[axis]) > 1e-9:
                    tc -= (p[axis] - value) / d1[axis]
            if not low < p[1 - axis] < high or abs(d1[axis]) <= 1e-9:
                continue
            speed = float(np.hypot(*d1))
            out = 1.0 if d1 @ np.array(OUTWARD[edge]) > 0 else -1.0
            tangent = out * d1 / speed
            local = [float(p[0] - x0), float(p[1] - y0)]
            local[axis] = float(value - (x0, y0)[axis])
            ends.append({"edge": edge, "ok": True, "curve": index, "x": round(local[0], 3), "y": round(local[1], 3),
                         "theta_deg": round(float(np.degrees(np.arctan2(tangent[1], tangent[0]))), 2),
                         "tangent": [round(float(tangent[0]), 4), round(float(tangent[1]), 4)],
                         "incidence_deg": round(float(np.degrees(np.arccos(np.clip(tangent @ np.array(OUTWARD[edge]), -1, 1)))), 1),
                         "curvature": round(float(out * (d1[0] * d2[1] - d1[1] * d2[0]) / speed ** 3), 5),
                         "width": curve.thickness, "colour": list(curve.colour), "t": round(float(tc), 6)})
    return ends


def save_json(path: Path, data, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent))


def next_puzzle_name(data: Path, name: str, n: int) -> str:
    """<name>_<n>x<n>_00000, with the next free number."""
    prefix = f"{name}_{n}x{n}_"
    taken = [int(p.name[len(prefix):]) for p in (data / "preprocessing").glob(prefix + "*") if p.name[len(prefix):].isdigit()]
    return f"{prefix}{max(taken, default=-1) + 1:05d}"


def make_puzzle(curves: list[Curve], picture: PictureParams, data: Path, name: str) -> Puzzle:
    """Draw the curves, cut the picture into pieces and write everything the solver and the evaluation read."""
    n, m, size = picture.pieces_per_side, picture.margin, picture.size
    xs = [round(c * size / n) for c in range(n + 1)]
    ys = [round(r * size / n) for r in range(n + 1)]
    if 2 * m >= min(xs[1] - xs[0], ys[1] - ys[0]):
        raise ValueError(f"margin {m} is too large for pieces of about {size // n} x {size // n} px")
    puzzle = next_puzzle_name(data, name, n)
    folder = data / "preprocessing" / puzzle
    for sub in ("images", "binary_masks"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    for c in curves:
        c.sample()
    img = draw(curves, size, size)
    img.save(folder / "full_image.png")

    truth, layout, count = {}, [], 0
    for r in range(n):
        for c in range(n):
            j = r * n + c
            x0, y0 = xs[c] + m, ys[r] + m
            w, h = xs[c + 1] - m - x0, ys[r + 1] - m - y0
            w, h = w | 1, h | 1
            box = (x0, y0, x0 + w, y0 + h)
            img.crop(box).convert("RGBA").save(folder / "images" / f"piece_{j:04d}.png")
            Image.new("L", (w, h), 255).save(folder / "binary_masks" / f"piece_{j:04d}.png")
            truth[f"piece_{j:04d}"] = {"translation": [-float(x0), -float(y0)], "rotation": 0}
            layout.append({"name": f"piece_{j:04d}", "row": r, "col": c, "box": box})
            ends = [e for k, cv in enumerate(curves) for e in exact_ends(cv, box, k)]
            ends.sort(key=lambda e: ("LRTB".index(e["edge"]), e["y"] if e["edge"] in "LR" else e["x"]))
            count += len(ends)
            save_json(data / "features" / puzzle / "exact" / f"piece_{j:04d}_ends.json",
                      {"piece": f"piece_{j:04d}", "size": [w, h], "ends": ends})

    gap = xs[1] - w
    save_json(folder / "ground_truth.json", truth, indent=3)
    save_json(folder / "layout.json", {"image_size": [size, size], "rows": n, "cols": n, "margin": m, "pieces": layout})
    save_json(folder / "curves_grid.json", {"piece_size": max(w, h), "xy_step": xs[1], "pieces_per_side": n})
    eq = [{"x": c.x, "y": c.y, "t": c.t, "thickness": c.thickness, "colour": list(c.colour)} for c in curves]
    save_json(folder / "equation.json", {**(eq[0] if len(eq) == 1 else {"curves": eq}), "size": [size, size], "gap": gap})
    what = f"x = {curves[0].x}   y = {curves[0].y}" if len(curves) == 1 else f"{len(curves)} curves"
    return Puzzle(puzzle, folder, n * n, (w, h), gap, count, what)
