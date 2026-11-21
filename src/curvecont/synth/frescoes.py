"""Fresco-style test pictures, after the two decors: Decor 1 (bands in a grid with a quarter circle in every panel
corner) and Decor 2 (a square with a circle, a turned square and four lobes; a frame; lunettes; rings).

Every picture is a list of curves (segments, arcs, ellipses) in the dataset format of synth/curves.py. The whole
decor is turned 10 to 30 degrees, so lines cross the cuts instead of running along them. With colour=True the same
shapes are painted in fresco colours by role (red-brown bands, blue or grey-green arcs, ochre circles, ...); the
colours come from a separate random stream, so the shapes stay the same with and without colour.
"""

from __future__ import annotations

import math
import random

SIZE = 600
STYLES = [["panels"], ["panels"], ["panels"], ["panels"], ["double_panels"], ["double_panels"], ["double_panels"],
          ["quatrefoil"], ["quatrefoil"], ["quatrefoil"], ["quatrefoil", "frame"], ["quatrefoil", "frame"], ["quatrefoil", "frame"],
          ["quatrefoil", "lunettes"], ["quatrefoil", "lunettes"], ["rings", "frame"], ["rings", "frame"],
          ["rings", "lunettes"], ["rings", "lunettes"], ["panels", "lunettes"]]
RED, BLUE, OCHRE, GREEN, DARK, ROSE = (160, 62, 45), (52, 112, 178), (206, 152, 48), (96, 132, 112), (64, 60, 64), (184, 90, 96)
CHOICES = {"band": [RED], "arc": [GREEN, BLUE], "square": [RED, DARK], "circle": [OCHRE], "diamond": [DARK, BLUE], "lobe": [BLUE, ROSE],
           "frame": [RED], "corner": [OCHRE, DARK], "lunette": [OCHRE, GREEN]}


def turn(theta: float, centre: tuple[float, float]):
    """Map decor coordinates (u, v) to picture coordinates: rotate by theta around the centre."""
    c, s = math.cos(theta), math.sin(theta)
    return lambda u, v: (centre[0] + c * u - s * v, centre[1] + s * u + c * v)


def segment(p, q, thickness, role) -> dict:
    return {"x": f"{p[0]:.1f} + ({q[0] - p[0]:.1f})*t", "y": f"{p[1]:.1f} + ({q[1] - p[1]:.1f})*t", "t": [0, 1],
            "thickness": thickness, "role": role}


def arc(centre, r, a0, a1, thickness, role) -> dict:
    return {"x": f"{centre[0]:.1f} + {r:.1f}*cos(t)", "y": f"{centre[1]:.1f} + {r:.1f}*sin(t)", "t": [round(a0, 4), round(a1, 4)],
            "thickness": thickness, "role": role}


def points(item: dict, n: int = 80) -> list[tuple[float, float]]:
    t0, t1 = item["t"]
    ts = [t0 + (t1 - t0) * k / (n - 1) for k in range(n)]
    names = {"cos": math.cos, "sin": math.sin}
    return [(eval(item["x"], names, {"t": t}), eval(item["y"], names, {"t": t})) for t in ts]


def visible(item: dict) -> bool:
    return any(10 < x < SIZE - 10 and 10 < y < SIZE - 10 for x, y in points(item))


def panels(rng: random.Random, centre, theta, double=False) -> list[dict]:
    """Decor 1: a grid of bands (single or double lines) with a quarter circle in every corner of every panel."""
    s = rng.uniform(230, 290)
    r = rng.uniform(0.30, 0.36) * s
    d = rng.uniform(22, 26) if double else 0.0
    th_band, th_arc = rng.choice([5, 6]), rng.choice([4, 5])
    P = turn(theta, centre)
    pu, pv = rng.uniform(0, s), rng.uniform(0, s)
    out = []
    for k in range(-3, 4):
        for off in ((-d / 2, d / 2) if double else (0.0,)):
            out.append(segment(P(k * s + pu + off, -460), P(k * s + pu + off, 460), th_band, "band"))
            out.append(segment(P(-460, k * s + pv + off), P(460, k * s + pv + off), th_band, "band"))
    delta = math.asin((d / 2 + 9) / r)  # arcs stop 9 px before the band
    for i in range(-3, 4):
        for j in range(-3, 4):
            node = P(i * s + pu, j * s + pv)
            out += [arc(node, r, theta + q * math.pi / 2 + delta, theta + (q + 1) * math.pi / 2 - delta, th_arc, "arc") for q in range(4)]
    return out


def quatrefoil(rng: random.Random, centre, theta) -> list[dict]:
    """Decor 2: a square with an inscribed circle, a turned square inside it, and a lobe at every corner."""
    a, th = rng.uniform(190, 240), rng.choice([5, 6])
    h, P = a / 2, turn(theta, centre)
    corners = [P(-h, -h), P(h, -h), P(h, h), P(-h, h)]
    out = [segment(corners[k], corners[(k + 1) % 4], th, "square") for k in range(4)]
    out.append({"circle": [round(centre[0], 1), round(centre[1], 1), round(0.4 * a, 1)], "thickness": th, "role": "circle"})
    rd = 0.4 * a * 0.78
    diamond = [P(rd * math.cos(k * math.pi / 2), rd * math.sin(k * math.pi / 2)) for k in range(4)]
    out += [segment(diamond[k], diamond[(k + 1) % 4], th - 1, "diamond") for k in range(4)]
    rho = rng.uniform(0.34, 0.42) * a
    for corner, deg in zip(corners, (225, 315, 45, 135), strict=True):
        mid = theta + math.radians(deg)
        out.append(arc(corner, rho, mid - 0.75 * math.pi, mid + 0.75 * math.pi, th, "lobe"))
    return out


def frame(rng: random.Random, centre, theta) -> list[dict]:
    """A double square frame with a small square in each inner corner."""
    F, d, th = rng.uniform(250, 275), rng.uniform(22, 26), rng.choice([6, 7])
    P = turn(theta, centre)
    out = []
    for h in (F, F - d):
        pts = [P(-h, -h), P(h, -h), P(h, h), P(-h, h)]
        out += [segment(pts[k], pts[(k + 1) % 4], th, "frame") for k in range(4)]
    q = rng.uniform(55, 75)
    for su in (-1, 1):
        for sv in (-1, 1):
            u0, v0 = su * (F - d), sv * (F - d)
            u1, v1 = u0 - su * q, v0 - sv * q
            out += [segment(P(u0, v1), P(u1, v1), th - 2, "corner"), segment(P(u1, v1), P(u1, v0), th - 2, "corner")]
    return out


def lunettes(rng: random.Random, centre, theta) -> list[dict]:
    """Two large arcs reaching into the picture from opposite sides."""
    R, depth, th = rng.uniform(330, 460), rng.uniform(100, 150), rng.choice([5, 6])
    P = turn(theta, centre)
    out = []
    for sgn in (-1, 1):
        c = P(0, sgn * (SIZE / 2 - depth + R))
        mid = theta - sgn * math.pi / 2
        out.append(arc(c, R, mid - 0.9, mid + 0.9, th, "lunette"))
    return out


def rings(rng: random.Random, centre, theta) -> list[dict]:
    """Two or three slightly oval rings inside each other."""
    r, th, out = rng.uniform(150, 180), rng.choice([5, 6]), []
    for k in range(rng.choice([2, 3])):
        a, b = r * rng.uniform(1.0, 1.1), r
        if a > 285:
            break
        c, s = math.cos(theta), math.sin(theta)
        out.append({"x": f"{centre[0]:.1f} + {a:.1f}*cos(t)*({c:.4f}) - {b:.1f}*sin(t)*({s:.4f})",
                    "y": f"{centre[1]:.1f} + {a:.1f}*cos(t)*({s:.4f}) + {b:.1f}*sin(t)*({c:.4f})", "t": [0, round(2 * math.pi, 6)],
                    "thickness": th, "role": f"ring{k}"})
        r = a + rng.uniform(32, 45)
    return out


MOTIFS = {"panels": panels, "double_panels": lambda rng, centre, theta: panels(rng, centre, theta, double=True),
          "quatrefoil": quatrefoil, "frame": frame, "lunettes": lunettes, "rings": rings}


def fresco_pictures(n: int = 20, seed: int = 1, colour: bool = False) -> list[dict]:
    """n pictures (dataset entries: name, style, many) cycling through the 20 styles."""
    rng, paint = random.Random(seed), random.Random(seed + 7919)
    pictures = []
    for k in range(n):
        style = STYLES[k % len(STYLES)]
        centre = (SIZE / 2 + rng.uniform(-35, 35), SIZE / 2 + rng.uniform(-35, 35))
        theta = rng.choice([-1, 1]) * math.radians(rng.uniform(10, 30))
        curves = [c for motif in style for c in MOTIFS[motif](rng, centre, theta) if "circle" in c or visible(c)]
        colours = {role: paint.choice(options) for role, options in CHOICES.items()}
        colours.update(zip(("ring0", "ring1", "ring2"), paint.sample([RED, BLUE, OCHRE], 3), strict=True))
        for c in curves:
            role = c.pop("role")
            if colour:
                c["colour"] = list(colours[role])
        pictures.append({"name": f"{'colour' if colour else 'fresco'}_{k + 1:02d}", "style": " + ".join(style), "many": curves})
    return pictures
