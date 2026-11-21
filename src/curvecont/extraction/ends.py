"""Find where painted curves leave a piece, and measure each end.

Steps for one piece image:
 1. contrast against the local background (median filter), and a mask of the paint;
 2. centre-line points of the strokes (ridge of the contrast, sub-pixel), linked into a graph;
 3. for every run of paint touching the border: follow the centre line inwards (geodesic distance), fit x(s), y(s)
    polynomials, and intersect them with the border;
 4. at that point: position, outward direction, incidence to the border normal, curvature (1/px, looking outwards),
    stroke width (full width at half contrast), colour (median along the centre line), and reliability flags:
    merged (the width or colour changes along the end: two strokes meet near the border, so width and colour are not
    trusted) and curvature_reliable (enough curve, and a close polynomial fit).
Ends that cannot be fitted keep ok = False and a note saying why.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from ..config import ExtractionParams

OUTWARD = {"L": (-1.0, 0.0), "R": (1.0, 0.0), "T": (0.0, -1.0), "B": (0.0, 1.0)}


def contrast_map(image: Image.Image, window: int) -> np.ndarray:
    """Colour distance of every pixel to its local background (median over a window)."""
    a = np.asarray(image.convert("RGB"), dtype=float)
    background = ndi.median_filter(a, size=(window, window, 1))
    return np.linalg.norm(a - background, axis=2)


def paint_mask(cmap: np.ndarray, min_contrast: float, min_size: int = 20) -> tuple[np.ndarray, float]:
    """Pixels with at least half the typical paint contrast, without specks; level 0 means no paint."""
    top = float(cmap.max())
    if top < min_contrast:
        return np.zeros(cmap.shape, bool), 0.0
    level = float(np.median(cmap[cmap > top / 2]))
    mask = cmap > level / 2
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = ndi.sum(mask, labels, np.arange(1, n + 1))
    mask = np.isin(labels, 1 + np.flatnonzero(sizes >= min_size))
    return (mask, level) if mask.any() else (mask, 0.0)


def stroke_width(mask: np.ndarray) -> float:
    edt = ndi.distance_transform_edt(mask)
    return 2.0 * float(np.percentile(edt[mask], 95)) - 0.5


def ridge_points(cmap: np.ndarray, mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel centre-line points of the strokes and the normal there (Hessian ridge detection)."""
    g = lambda order: ndi.gaussian_filter(cmap, sigma, order=order, mode="nearest")  # noqa: E731
    fx, fy, fxx, fxy, fyy = g((0, 1)), g((1, 0)), g((0, 2)), g((1, 1)), g((2, 0))
    phi = 0.5 * np.arctan2(2 * fxy, fxx - fyy)
    nx, ny = -np.sin(phi), np.cos(phi)
    lam = 0.5 * (fxx + fyy) - np.hypot(0.5 * (fxx - fyy), fxy)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(fx * nx + fy * ny) / lam
    inside = 0.6
    ok = mask & (lam < 0) & (np.abs(t * nx) <= inside) & (np.abs(t * ny) <= inside)
    ys, xs = np.nonzero(ok)
    pts = np.c_[xs + t[ok] * nx[ok] + 0.5, ys + t[ok] * ny[ok] + 0.5]
    return pts, np.c_[nx[ok], ny[ok]]


def link(pts: np.ndarray, nrm: np.ndarray, radius: float = 2.2, max_turn_deg: float = 40.0):
    """Graph of neighbouring centre-line points with similar normals (sparse, distances as weights)."""
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    step = pts[j] - pts[i]
    dist = np.maximum(np.hypot(step[:, 0], step[:, 1]), 1e-6)
    same_direction = np.abs((nrm[i] * nrm[j]).sum(1)) >= np.cos(np.radians(max_turn_deg))
    return coo_matrix((dist[same_direction], (i[same_direction], j[same_direction])), shape=(len(pts),) * 2).tocsr()


def border_runs(mask: np.ndarray) -> list[tuple[str, int, int]]:
    """Runs of paint along each border: (edge, first, last) pixel along the edge."""
    strips = {"L": mask[:, 0], "R": mask[:, -1], "T": mask[0, :], "B": mask[-1, :]}
    runs = []
    for edge, line in strips.items():
        change = np.diff(np.r_[False, line, False].astype(int))
        runs += [(edge, int(a), int(b) - 1) for a, b in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1), strict=True)]
    return runs


def width_along(cmap: np.ndarray, pts: np.ndarray, nrm: np.ndarray, reach: float) -> np.ndarray:
    """Stroke width (full width at half the peak contrast) across the centre line at every point."""
    step = 0.25
    t = np.arange(-reach, reach + 1e-9, step)
    xs = pts[:, :1] - 0.5 + nrm[:, :1] * t
    ys = pts[:, 1:] - 0.5 + nrm[:, 1:] * t
    v = ndi.map_coordinates(cmap, [ys, xs], order=1, mode="nearest")
    half = 0.5 * v.max(axis=1)

    def half_distance(side):
        below = side < half[:, None]
        k = below.argmax(axis=1)
        found = below.any(axis=1) & (k > 0)
        k = np.maximum(k, 1)
        a, b = np.take_along_axis(side, (k - 1)[:, None], 1)[:, 0], np.take_along_axis(side, k[:, None], 1)[:, 0]
        frac = (a - half) / np.where(a == b, 1.0, a - b)
        return np.where(found, (k - 1 + frac) * step, np.nan)

    centre = len(t) // 2
    return half_distance(v[:, centre:]) + half_distance(v[:, centre::-1])


def fit_window(s: np.ndarray, P: np.ndarray, length: float, degree: int) -> dict | None:
    """Fit x(s) and y(s) to the centre-line points within geodesic distance `length`, dropping outliers once."""
    sel = np.flatnonzero(s <= length)
    S, X, Y = s[sel], P[sel, 0], P[sel, 1]
    deg = degree if np.ptp(S) >= 25 else min(degree, 2)
    if len(S) < deg + 4:
        return None
    fx, fy = Polynomial.fit(S, X, deg), Polynomial.fit(S, Y, deg)
    r = np.hypot(X - fx(S), Y - fy(S))
    keep = r <= max(0.25, 4.45 * np.median(r))
    if not keep.all() and keep.sum() >= deg + 4:
        sel, S, X, Y = sel[keep], S[keep], X[keep], Y[keep]
        fx, fy = Polynomial.fit(S, X, deg), Polynomial.fit(S, Y, deg)
        r = np.hypot(X - fx(S), Y - fy(S))
    return {"sel": sel, "S": S, "fx": fx, "fy": fy, "rms": float(np.sqrt(np.mean(r ** 2)))}

ALONG = {"L": (0.0, 1.0), "R": (0.0, 1.0), "T": (1.0, 0.0), "B": (1.0, 0.0)}  # unit vector along each edge


def end_from_contact(end: dict, edge: str, first: int, last: int, region: np.ndarray, rgb: np.ndarray, reach: float,
                     params: ExtractionParams) -> dict:
    """An end measured from its paint contact alone, for a stub too small for a centre line. A stroke of width w
    crossing the border at beta from the normal touches it over w / cos(beta), so the contact length gives beta; the
    side on which the paint lies gives the lean, and the middle of the contact the position. No curvature, and the
    width (the piece's typical stroke width) is marked unreliable."""
    H, W = region.shape
    n, b = np.array(OUTWARD[edge]), np.array(ALONG[edge])
    c = (first + last + 1) / 2.0
    mid = np.array({"L": (0.5, c), "R": (W - 0.5, c), "T": (c, 0.5), "B": (c, H - 0.5)}[edge])  # on the outer pixel row
    ys, xs = np.nonzero(region)
    near = np.hypot(xs + 0.5 - mid[0], ys + 0.5 - mid[1]) <= 2 * reach
    ys, xs = ys[near], xs[near]
    width = reach / 2  # the piece's typical stroke width
    beta = float(np.arccos(min(1.0, width / (last + 1 - first))))
    incidence = float(np.degrees(beta))
    if incidence > params.max_incidence_deg:
        end["note"] = f"curve meets the border at a very shallow angle ({incidence:.0f} deg from the normal)"
        return end
    s = 1.0 if (mid - np.array([xs.mean() + 0.5, ys.mean() + 0.5])) @ b >= 0 else -1.0
    tangent = np.cos(beta) * n + s * np.sin(beta) * b
    x_end, y_end = mid + 0.5 * n + 0.5 * np.tan(beta) * s * b  # from the outer pixel row out to the border line
    colour = np.median(rgb[ys, xs], axis=0)
    end.update(ok=True, x=round(float(x_end), 3), y=round(float(y_end), 3),
               theta_deg=round(float(np.degrees(np.arctan2(tangent[1], tangent[0]))), 2),
               tangent=[round(float(tangent[0]), 4), round(float(tangent[1]), 4)], incidence_deg=round(incidence, 1),
               curvature=None, width=round(width, 2), colour=[int(round(v)) for v in colour], n_points=0, merged=False,
               width_reliable=False, colour_reliable=True, curvature_reliable=False, from_contact=True)
    return end

def fit_end(edge, first, last, P, N, lab_pts, labels, graph, cmap, guard, reach, rgb, params: ExtractionParams) -> dict:
    """Measure the curve end that touches `edge` between pixels first and last (see the module text)."""
    H, W = cmap.shape
    end = {"edge": edge, "contact": [first, last + 1], "ok": False}
    c = (first + last + 1) / 2.0
    p0 = np.array({"L": (0.0, c), "R": (W, c), "T": (c, 0.0), "B": (c, H)}[edge], float)
    row, col = {"L": (first, 0), "R": (first, W - 1), "T": (0, first), "B": (H - 1, first)}[edge]
    cand = np.flatnonzero(lab_pts == labels[row, col])
    if cand.size < 8 or graph is None:  # a stub too small for a centre line: measure it from its paint contact
        return end_from_contact(end, edge, first, last, labels == labels[row, col], rgb, reach, params)
    d0 = np.hypot(P[cand, 0] - p0[0], P[cand, 1] - p0[1])
    src = int(cand[np.argmin(d0)])
    if d0.min() > 3 * (guard + last - first + 1) + 10:
        end["note"] = "centre line too far from the border"
        return end
    s = dijkstra(graph, directed=False, indices=src)
    total = float(s[np.isfinite(s)].max())
    if total < 5:
        end["note"] = "curve too short"
        return end
    short = total < 15  # a short stub: a straight line through all its points still gives position and direction
    if short:
        fits = [fit_window(s, P, total, 1)]
    else:
        fits = [fit_window(s, P, min(params.length * k, 0.6 * total), params.degree) for k in (1.0, 0.75, 0.5, 1 / 3)]
    fits = [f for f in fits if f is not None]
    if not fits:
        end["note"] = "too few points"
        return end
    limit = max(0.03, 1.5 * fits[-1]["rms"])
    best = next((f for f in fits if f["rms"] <= limit), fits[-1])
    sel, S, fx, fy = best["sel"], best["S"], best["fx"], best["fy"]
    along_x = edge in "LR"
    f, edge_value = (fx, 0.0 if edge == "L" else float(W)) if along_x else (fy, 0.0 if edge == "T" else float(H))
    roots = (f - edge_value).roots()
    roots = roots[np.abs(roots.imag) < 1e-9].real
    if roots.size == 0:
        end["note"] = "fit does not reach the border"
        return end
    s0 = float(roots[np.argmin(np.abs(roots))])
    if abs(s0) > 4 * guard + 20:
        end["note"] = "fit does not reach the border"
        return end
    dx, dy = fx.deriv(1)(s0), fy.deriv(1)(s0)
    ddx, ddy = fx.deriv(2)(s0), fy.deriv(2)(s0)
    speed = float(np.hypot(dx, dy))
    tangent = -np.array([dx, dy]) / speed
    incidence = float(np.degrees(np.arccos(np.clip(tangent @ np.array(OUTWARD[edge]), -1, 1))))
    if incidence > params.max_incidence_deg:
        end["note"] = f"curve meets the border at a very shallow angle ({incidence:.0f} deg from the normal)"
        return end
    arc = float(np.ptp(S))
    kappa = -(dx * ddy - dy * ddx) / speed ** 3
    kappa = round(float(kappa), 5) if arc >= params.min_arc_for_curvature else None
    x_end, y_end = (edge_value, fy(s0)) if along_x else (fx(s0), edge_value)
    widths = width_along(cmap, P[sel], N[sel], reach)
    width = float(np.nanmedian(widths)) if np.isfinite(widths).any() else reach / 2
    width_spread = float((np.nanpercentile(widths, 90) - np.nanpercentile(widths, 10)) / width) if np.isfinite(widths).any() else 0.0
    px = np.clip(P[sel].astype(int), 0, [W - 1, H - 1])
    pixels = rgb[px[:, 1], px[:, 0]]
    colour = np.median(pixels, axis=0)
    colour_spread = float(np.mean(np.linalg.norm(pixels - colour, axis=1) > 40))
    merged = width_spread > params.max_width_spread or colour_spread > params.max_colour_spread
    end.update(ok=True, x=round(float(x_end), 3), y=round(float(y_end), 3),
               theta_deg=round(float(np.degrees(np.arctan2(tangent[1], tangent[0]))), 2),
               tangent=[round(float(tangent[0]), 4), round(float(tangent[1]), 4)],
               incidence_deg=round(incidence, 1), curvature=kappa, width=round(width, 2), colour=[int(round(v)) for v in colour],
               n_points=int(len(S)), arc_px=round(arc, 1), rms_px=round(best["rms"], 3),
               width_spread=round(width_spread, 3), colour_spread=round(colour_spread, 3), merged=merged,
               width_reliable=not merged, colour_reliable=not merged,
               curvature_reliable=kappa is not None and not merged and best["rms"] <= params.max_rms_for_curvature)
    if short:
        end["short_stub"] = True
    return end


def analyse(img: Image.Image, params: ExtractionParams) -> dict:
    """All curve ends of one piece image: {"level", "sigma", "guard", "ends": [...]}."""
    cmap = contrast_map(img, params.window)
    mask, level = paint_mask(cmap, params.min_contrast)
    H, W = cmap.shape
    res = {"level": level, "sigma": None, "guard": None, "ends": []}
    if level == 0:
        return res
    width = stroke_width(mask)
    sigma = float(params.sigma or np.clip(0.3 * width, 1.0, 6.0))
    guard = sigma
    pts, nrm = ridge_points(cmap, mask, sigma)
    res.update(sigma=sigma, guard=guard)
    border = np.minimum.reduce([pts[:, 0], W - pts[:, 0], pts[:, 1], H - pts[:, 1]]) if len(pts) else np.zeros(0)
    trusted = border >= guard
    P, N = pts[trusted], nrm[trusted]
    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    lab_pts = labels[np.minimum(P[:, 1].astype(int), H - 1), np.minimum(P[:, 0].astype(int), W - 1)] if len(P) else np.zeros(0, int)
    graph = link(P, N) if len(P) >= 8 else None
    reach = max(8.0, 2.0 * width)
    rgb = np.asarray(img.convert("RGB"), dtype=float)
    for edge, first, last in border_runs(mask):
        res["ends"].append(fit_end(edge, first, last, P, N, lab_pts, labels, graph, cmap, guard, reach, rgb, params))
    res["ends"].sort(key=lambda e: ("LRTB".index(e["edge"]), e["contact"][0]))
    return res


def extract_pieces(images: list[Path], out_dir: Path, params: ExtractionParams) -> str:
    """Write <piece>_ends.json for every image to out_dir; returns a one-line report (and why ends failed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    found = []
    for path in images:
        img = Image.open(path).convert("RGB")
        res = analyse(img, params)
        data = {"piece": path.stem, "image": path.as_posix(), "size": [img.width, img.height], "paint_level": round(res["level"], 2),
                "params": {"window": params.window, "sigma": None if res["sigma"] is None else round(res["sigma"], 2),
                           "length": params.length, "degree": params.degree,
                           "guard": None if res["guard"] is None else round(res["guard"], 2)},
                "ends": res["ends"]}
        (out_dir / f"{path.stem}_ends.json").write_text(json.dumps(data, indent=2))
        found.append(res["ends"])
    ends = [e for piece in found for e in piece]
    failed = Counter(e["note"].split(" (")[0] for e in ends if not e["ok"])
    report = (f"{len(found)} pieces ({sum(not piece for piece in found)} without a curve end), {len(ends)} curve ends, "
              f"{len(ends) - sum(failed.values())} fitted, {sum(failed.values())} failed")
    if failed:
        report += "\n  failed: " + ", ".join(f"{reason} {n}" for reason, n in failed.most_common())
    return report
