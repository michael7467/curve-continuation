"""Found ends (extraction) against exact ends (equations).

Each exact end is matched to the nearest unused found end on the same edge within the tolerance (greedy, one to one).
The result is a table with one row per exact end (position, direction, curvature, width and colour errors) and a
short report.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..scoring.features import lab

COLUMNS = ["piece", "edge", "curve", "found", "exact_x", "exact_y", "exact_theta_deg", "exact_curvature", "exact_width",
           "exact_colour", "x", "y", "theta_deg", "curvature", "width", "colour", "merged", "curvature_reliable", "position_error_px",
           "direction_error_deg", "curvature_error", "width_error_px", "colour_error"]


def hex_colour(c) -> str:
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" if c else ""


def compare_ends(exact_dir: Path, found_dir: Path, tolerance: float) -> tuple[list[dict], int]:
    """Rows (one per exact end) and the number of extra found ends that match no exact end."""
    exact_files = sorted(Path(exact_dir).glob("*_ends.json"))
    if not exact_files:
        raise FileNotFoundError(f"no exact ends in {Path(exact_dir).as_posix()}")
    rows, extra = [], 0
    for f in exact_files:
        g = Path(found_dir) / f.name
        if not g.exists():
            raise FileNotFoundError(f"{g.as_posix()} not found: extract the curve ends first")
        exact = json.loads(f.read_text())["ends"]
        found = [e for e in json.loads(g.read_text())["ends"] if e["ok"]]
        used = set()
        for e in exact:
            row = {"piece": f.name[:-10], "edge": e["edge"], "curve": e.get("curve", 0), "found": 0, "exact_x": e["x"],
                   "exact_y": e["y"], "exact_theta_deg": e["theta_deg"], "exact_curvature": e["curvature"],
                   "exact_width": e.get("width"), "exact_colour": hex_colour(e.get("colour"))}
            near = [(np.hypot(m["x"] - e["x"], m["y"] - e["y"]), i) for i, m in enumerate(found)
                    if m["edge"] == e["edge"] and i not in used]
            if near and min(near)[0] <= tolerance:
                d, i = min(near)
                m = found[i]
                used.add(i)
                row.update(found=1, x=m["x"], y=m["y"], theta_deg=m["theta_deg"], curvature=m["curvature"],
                           position_error_px=round(float(d), 3),
                           direction_error_deg=round((m["theta_deg"] - e["theta_deg"] + 180) % 360 - 180, 2))
                if m["curvature"] is not None:
                    row["curvature_error"] = round(m["curvature"] - e["curvature"], 5)
                row.update(merged=int(m.get("merged", False)),
                           curvature_reliable=int(m.get("curvature_reliable", m["curvature"] is not None)))
                if m.get("width") is not None and e.get("width") is not None:
                    row.update(width=m["width"], width_error_px=round(m["width"] - e["width"], 2))
                if m.get("colour") and e.get("colour"):
                    difference = float(np.linalg.norm(lab(m["colour"]) - lab(e["colour"])))
                    row.update(colour=hex_colour(m["colour"]), colour_error=round(difference, 2))
            rows.append(row)
        extra += len(found) - len(used)
    return rows, extra


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict], extra: int, label: str) -> list[str]:
    """The text report: ends found, by curve, and the measurement errors."""
    found = [r for r in rows if r["found"]]
    lines = [f"{label}: {len(rows)} exact ends, {len(found)} found by extraction, {len(rows) - len(found)} missed, {extra} extra"]
    ids = sorted({r["curve"] for r in rows})
    if len(ids) > 1:
        lines.append("  by curve: " + ", ".join(f"curve {k}: {sum(r['found'] for r in rows if r['curve'] == k)} / "
                                                f"{sum(r['curve'] == k for r in rows)}" for k in ids))
    if not found:
        return lines
    pos = [r["position_error_px"] for r in found]
    ang = [abs(r["direction_error_deg"]) for r in found]
    cur = [abs(r["curvature_error"]) for r in found if "curvature_error" in r]
    lines.append(f"  position error: mean {np.mean(pos):.3f} px, max {np.max(pos):.3f} px")
    lines.append(f"  direction error: mean {np.mean(ang):.2f} deg, max {np.max(ang):.2f} deg")
    if cur:
        lines.append(f"  curvature error: mean {np.mean(cur):.5f}, max {np.max(cur):.5f} "
                     f"(exact curvature up to {max(abs(r['exact_curvature']) for r in found):.5f})")
    clean = [r for r in found if not r.get("merged")]
    wid = [abs(r["width_error_px"]) for r in clean if "width_error_px" in r]
    col = [r["colour_error"] for r in clean if "colour_error" in r]
    merged = len(found) - len(clean)
    if wid:
        lines.append(f"  width error: mean {np.mean(wid):.2f} px, max {np.max(wid):.2f} px"
                     + (" (ends not marked merged)" if merged else ""))
    if col:
        lines.append(f"  colour error: mean {np.mean(col):.1f}, max {np.max(col):.1f} (Lab difference, 2 is barely visible"
                     + ("; ends not marked merged)" if merged else ")"))
    if any("merged" in r for r in found):
        unsure = sum(1 for r in found if r.get("curvature") is not None and not r.get("curvature_reliable"))
        lines.append(f"  marked by extraction: {merged} merged ends (width and colour not used), {unsure} curvatures not trusted")
    return lines
