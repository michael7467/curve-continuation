import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

TAU = 1.0
FACING = {"right": ("R", "L"), "left": ("L", "R"), "top": ("T", "B"), "bottom": ("B", "T")}
ORDER = ("left", "top", "right", "bottom")
DIAGONAL = ("top-left", "top-right", "bottom-left", "bottom-right")
STEP_OF = {"right": (0, 1), "left": (0, -1), "bottom": (1, 0), "top": (-1, 0),
           "top-left": (-1, -1), "top-right": (-1, 1), "bottom-left": (1, -1), "bottom-right": (1, 1)}
STEP = {v: k for k, v in STEP_OF.items() if k in FACING}
SIDE_OF = {"left": "L", "right": "R", "top": "T", "bottom": "B"}
DIAG_FACING = {"top-left": (("L", "T"), ("R", "B")), "top-right": (("R", "T"), ("L", "B")),
               "bottom-left": (("L", "B"), ("R", "T")), "bottom-right": (("R", "B"), ("L", "T"))}
DIAG_OF = {"right": ("top-right", "bottom-right"), "left": ("top-left", "bottom-left"),
           "top": ("top-left", "top-right"), "bottom": ("bottom-left", "bottom-right")}
OPPOSITE = {"top-left": "bottom-right", "bottom-right": "top-left", "top-right": "bottom-left", "bottom-left": "top-right"}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def arc_over_chord(phi):
    half = phi / 2
    return 1.0 if abs(half) < 5e-7 else half / np.sin(half)


def biarc_closed(P0, th0, P1, th1):
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    t1, t2 = np.array([np.cos(th0), np.sin(th0)]), np.array([np.cos(th1), np.sin(th1)])
    v = P1 - P0
    vv = float(v @ v)
    if vv < 1e-12:
        return None
    one_minus_dot = 2.0 * np.sin(wrap(th1 - th0) / 2) ** 2
    b = float(v @ (t1 + t2))
    if one_minus_dot < 1e-12:
        joint = (P0 + P1) / 2
    else:
        s = np.sqrt(b * b + 2.0 * one_minus_dot * vv)
        d = vv / (b + s) if b >= 0 else (s - b) / (2.0 * one_minus_dot)
        joint = (P0 + P1 + d * (t1 - t2)) / 2
    w1, w2 = joint - P0, P1 - joint
    phi1 = 2 * wrap(np.arctan2(w1[1], w1[0]) - th0)
    phi2 = 2 * wrap(th1 - np.arctan2(w2[1], w2[0]))
    if min(np.pi - abs(phi1 / 2), np.pi - abs(phi2 / 2)) < 1e-9:
        return None
    L1, L2 = np.hypot(w1[0], w1[1]) * arc_over_chord(phi1), np.hypot(w2[0], w2[1]) * arc_over_chord(phi2)
    if min(L1, L2) < 1e-9:
        return None
    bending = phi1 ** 2 / L1 + phi2 ** 2 / L2
    if not np.isfinite(bending):
        return None
    joint_angle = float(wrap(th0 + phi1))
    return {"cost": float(bending * (L1 + L2)),
            "arcs": [(P0, th0, float(phi1), float(L1)), (joint, joint_angle, float(phi2), float(L2))]}


def sample_arc(P, th, phi, L, n=80):
    u = np.linspace(0, 1, n)
    if abs(phi) < 1e-9:
        return P + L * u[:, None] * np.array([np.cos(th), np.sin(th)])
    R, ang = L / phi, th + phi * u
    return np.c_[P[0] + R * (np.sin(ang) - np.sin(th)), P[1] - R * (np.cos(ang) - np.cos(th))]


def score_from_cost(cost, tau):
    return float(1.0 - min(cost, tau) / tau)


def offset(shape_a, shape_b, where, gap):
    (ha, wa), (hb, wb) = shape_a, shape_b
    return np.array({"right": (wa + gap, 0), "left": (-(wb + gap), 0),
                     "bottom": (0, ha + gap), "top": (0, -(hb + gap)),
                     "bottom-right": (wa + gap, ha + gap), "bottom-left": (-(wb + gap), ha + gap),
                     "top-right": (wa + gap, -(hb + gap)), "top-left": (-(wb + gap), -(hb + gap))}[where], float)


def leaves_gaps(bi, shape_a, shape_b, shift, slack):
    (ha, wa), (hb, wb) = shape_a, shape_b
    sx, sy = float(shift[0]), float(shift[1])
    cells = [(0.0, wa, 0.0, ha), (sx, sx + wb, sy, sy + hb), (sx, sx + wb, 0.0, ha), (0.0, wa, sy, sy + hb)]
    pts = np.vstack([sample_arc(P, th, phi, L) for P, th, phi, L in bi["arcs"]])
    for x0, x1, y0, y1 in cells:
        depth = np.minimum.reduce([pts[:, 0] - x0, x1 - pts[:, 0], pts[:, 1] - y0, y1 - pts[:, 1]])
        if depth.max() > slack:
            return True
    return False


def join_cost(A, B, edge_a, edges_b, where, shift):
    costs = []
    for a in (e for e in A["ends"] if e["edge"] == edge_a):
        for b in (e for e in B["ends"] if e["edge"] in edges_b):
            bi = biarc_closed(a["pos"], a["theta"], b["pos"] + shift, b["theta"] + np.pi)
            cut = where in DIAGONAL and bi is not None and leaves_gaps(bi, A["shape"], B["shape"], shift, (a["width"] + b["width"]) / 2)
            costs.append(np.inf if bi is None or cut else bi["cost"])
    return min(costs) if costs else None


def evaluate_pair(A, B, gap, diagonal):
    out = {}
    for where in list(FACING) + (list(DIAGONAL) if diagonal else []):
        edges_a, edges_b = ((FACING[where][0],), (FACING[where][1],)) if where in FACING else DIAG_FACING[where]
        shift = offset(A["shape"], B["shape"], where, gap)
        costs = {edge_a: join_cost(A, B, edge_a, edges_b, where, shift) for edge_a in edges_a}
        found = [c for c in costs.values() if c is not None]
        out[where] = min(found) if found else None
        if where in DIAGONAL:
            for edge_a, c in costs.items():
                out[(where, edge_a)] = c
    return out


def true_position(place, a, b):
    (pa, ra, ca), (pb, rb, cb) = place[a], place[b]
    return STEP.get((rb - ra, cb - ca)) if pa == pb else None


def true_partner(pieces, place, cell, a, where, diagonal):
    image, r, c = place[a]
    dr, dc = STEP_OF[where]
    straight = cell.get((image, r + dr, c + dc))
    if straight is None:
        return None
    if not diagonal or any(e["edge"] == FACING[where][1] for e in pieces[straight]["ends"]):
        return straight, where, "straight"
    end = next(e for e in pieces[a]["ends"] if e["edge"] == SIDE_OF[where])
    h, w = pieces[a]["shape"]
    along = end["pos"][1] / h if where in ("left", "right") else end["pos"][0] / w
    pos = DIAG_OF[where][0 if along < 0.5 else 1]
    ddr, ddc = STEP_OF[pos]
    d = cell.get((image, r + ddr, c + ddc))
    if d is not None and any(e["edge"] in DIAG_FACING[pos][1] for e in pieces[d]["ends"]):
        return d, pos, "diagonal"
    return straight, where, "none"


def diagonal_pairs(questions):
    pairs = {}
    for q in questions:
        if q["kind"] == "diagonal":
            pairs[(q["piece"], q["true_neighbour"])] = q["true_where"]
            pairs[(q["true_neighbour"], q["piece"])] = OPPOSITE[q["true_where"]]
    return pairs


def candidate_costs(placements, names, a, where, diagonal):
    cost = {}
    for b in names:
        if b == a:
            continue
        res = placements[(a, b)]
        cost[(b, where)] = np.inf if res[where] is None else res[where]
        if diagonal:
            for pos in DIAG_OF[where]:
                r = res[(pos, SIDE_OF[where])]
                cost[(b, pos)] = np.inf if r is None else r
    return cost


def run_tests(pieces, place, gap, diagonal):
    names = list(pieces)
    placements = {(a, b): evaluate_pair(pieces[a], pieces[b], gap, diagonal) for a in names for b in names if a != b}
    cell = {v: k for k, v in place.items()}
    questions = []
    for a in names:
        for where in ORDER:
            if not any(e["edge"] == SIDE_OF[where] for e in pieces[a]["ends"]):
                continue
            partner = true_partner(pieces, place, cell, a, where, diagonal)
            if partner is None:
                continue
            truth, truth_where, kind = partner
            cost = candidate_costs(placements, names, a, where, diagonal)
            key = (truth, truth_where)
            wrong = sorted((k for k in cost if k != key), key=cost.get)
            best_wrong = wrong[0] if wrong and np.isfinite(cost[wrong[0]]) else None
            best_wrong_cost = cost[wrong[0]] if wrong else np.inf
            true_cost = cost[key]
            rank = 1 + sum(1 for k in wrong if cost[k] < true_cost) if np.isfinite(true_cost) else None
            questions.append({"piece": a, "side": SIDE_OF[where], "true_neighbour": truth, "true_where": truth_where,
                              "kind": kind, "rank": rank, "true_cost": true_cost,
                              "best_wrong": best_wrong[0] if best_wrong else None,
                              "best_wrong_where": best_wrong[1] if best_wrong else None,
                              "best_wrong_cost": best_wrong_cost,
                              "candidates": int(sum(np.isfinite(v) for v in cost.values())),
                              "correct": bool(rank == 1 and best_wrong_cost > true_cost)})
    return placements, questions


def position_test(names, place, placements, questions, tau, diagonal):
    places = ORDER + DIAGONAL if diagonal else ORDER
    dpairs = diagonal_pairs(questions) if diagonal else {}
    won, total = [0, 0], [0, 0]
    for a in names:
        for b in names:
            if a == b:
                continue
            res = placements[(a, b)]
            good, kind = true_position(place, a, b), 0
            if good is None and (a, b) in dpairs:
                good, kind = dpairs[(a, b)], 1
            if good is None or res[good] is None:
                continue
            score = {w: 0.0 if res[w] is None else score_from_cost(res[w], tau) for w in places}
            total[kind] += 1
            won[kind] += score[good] > 0 and score[good] > max(v for w, v in score.items() if w != good)
    return won, total


def read_piece(ends_dir, pieces_dir, name):
    f = ends_dir / f"{name}_ends.json"
    if not f.exists():
        raise SystemExit(f"{f.as_posix()} not found. Run  python src/extract.py {pieces_dir.as_posix()}  first.")
    d = json.loads(f.read_text())
    W, H = d["size"]
    ends = [{"edge": e["edge"], "pos": np.array([e["x"], e["y"]], float), "theta": float(np.radians(e["theta_deg"])),
             "width": float(e.get("width", 6.0))} for e in d["ends"] if e["ok"]]
    return {"shape": (H, W), "ends": ends}


def read_folder(folder, ends_root):
    layout_file = folder / "layout.json"
    if not layout_file.exists():
        raise SystemExit(f"{layout_file.as_posix()} not found: cut_pieces.py or pool_pieces.py writes it (the margin, and which piece came from where).")
    layout = json.loads(layout_file.read_text())
    place = {p["name"]: (p.get("puzzle", ""), p["row"], p["col"]) for p in layout["pieces"]}
    ends_dir = Path(ends_root) / folder.resolve().name
    return layout, place, {p["name"]: read_piece(ends_dir, folder, p["name"]) for p in layout["pieces"]}


def cost_text(x):
    return "" if not np.isfinite(x) else f"{x:.6f}"


def save_questions(path, questions, diagonal):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["piece", "side", "true_neighbour", "rank_of_true", "true_cost", "best_wrong", "best_wrong_cost", "candidates", "correct"]
                        + (["true_where", "kind", "best_wrong_where"] if diagonal else []))
        for q in questions:
            writer.writerow([q["piece"], q["side"], q["true_neighbour"], "" if q["rank"] is None else q["rank"],
                             cost_text(q["true_cost"]), q["best_wrong"] or "", cost_text(q["best_wrong_cost"]),
                             q["candidates"], int(q["correct"])]
                            + ([q["true_where"], q["kind"], q["best_wrong_where"] or ""] if diagonal else []))


def process(folder, a):
    layout, place, pieces = read_folder(folder, a.ends)
    gap = float(a.gap if a.gap is not None else 2 * layout["margin"])
    if gap < 2:
        print(f"WARNING {folder.as_posix()}: a gap of {gap:g} px is too small for a biarc. Cut again with  --margin 8  (gap 16 px).")
    placements, questions = run_tests(pieces, place, gap, a.diagonal)
    won, total = position_test(list(pieces), place, placements, questions, a.tau, a.diagonal)
    ranks = [q["rank"] for q in questions if q["rank"] is not None]
    first = sum(q["correct"] for q in questions)
    out = Path(a.out) / folder.resolve().name
    out.mkdir(parents=True, exist_ok=True)
    save_questions(out / "questions.csv", questions, a.diagonal)
    print(f"{folder.as_posix()}: {len(pieces)} pieces, gap {gap:g} px, tau {a.tau:g}")
    who = "partner" if a.diagonal else "neighbour"
    print(f"  {who} question: true {who} first in {first} / {len(questions)}, "
          f"top 3 in {sum(r <= 3 for r in ranks)}, worst rank {max(ranks, default=None)}, {len(questions) - len(ranks)} without score")
    if a.diagonal:
        print(f"  eight-position test: true position best in {won[0]} / {total[0]} straight pairs and {won[1]} / {total[1]} diagonal pairs")
    else:
        print(f"  four-position test: true position best in {won[0]} / {total[0]} pairs")
    return first, len(questions), sum(won), sum(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="a folder of mixed pieces such as data/pieces/all_3x3, or a pattern")
    ap.add_argument("--ends", default="results/extract", help="where extract.py wrote its files (default results/extract)")
    ap.add_argument("--diagonal", action="store_true",
                    help="also try the four diagonal positions: a curve that leaves a piece through a corner continues in the diagonal piece "
                         "(the joining curve must stay inside the gaps at the corner)")
    ap.add_argument("--out", default=None, help="output folder for questions.csv (default results/biarc_closed; _diagonal is added with --diagonal)")
    ap.add_argument("--tau", type=float, default=TAU, help="cost in rad^2 at which the score reaches 0 (default 1.0)")
    ap.add_argument("--gap", type=float, help="gap between neighbouring pieces in px (default: 2 x margin from layout.json)")
    a = ap.parse_args()
    if a.out is None:
        a.out = "results/biarc_closed" + ("_diagonal" if a.diagonal else "")
    folders = []
    for item in a.folders:
        folders += [Path(m) for m in sorted(glob.glob(item)) or [item] if Path(m).is_dir()]
    if not folders:
        raise SystemExit("No folder found. Give a folder of pieces such as data/pieces/all_3x3, or a pattern.")
    results = [process(f, a) for f in folders]
    if len(results) > 1:
        first, asked, won, pairs = (sum(r[i] for r in results) for i in range(4))
        print(f"ALL {len(results)} folders: true answer first in {first} / {asked} questions, true position best in {won} / {pairs} pairs")


if __name__ == "__main__":
    main()
