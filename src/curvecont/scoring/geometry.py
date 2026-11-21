"""The cost of joining two curve ends with a smooth curve.

Every cost takes P0 with direction th0 (leaving the first end) and P1 with direction th1 (arriving at the second end),
positions in px and angles in rad, and returns a dimensionless cost (0 = perfect continuation) or None when the
geometry is degenerate. The biarc and clothoid costs are both length x integral of curvature^2 along the joining
curve, so they agree on circles (the turning angle squared) and on straight lines (0).
"""

from __future__ import annotations

import numpy as np


def wrap(a):
    """Angle in [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def arc_over_chord(phi: float) -> float:
    """Length of a circular arc turning by phi divided by its chord."""
    half = phi / 2
    return 1.0 if abs(half) < 5e-7 else half / np.sin(half)


def biarc(P0, th0, P1, th1) -> dict | None:
    """The biarc (two circular arcs meeting with the same tangent) from (P0, th0) to (P1, th1), with the joint chosen
    on the standard family. Returns {"cost": (phi1^2/L1 + phi2^2/L2)(L1 + L2), "arcs": [(start, direction, turn, length)]}."""
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
    L1, L2 = np.hypot(*w1) * arc_over_chord(phi1), np.hypot(*w2) * arc_over_chord(phi2)
    if min(L1, L2) < 1e-9:
        return None
    return {"cost": float((phi1 ** 2 / L1 + phi2 ** 2 / L2) * (L1 + L2)),
            "arcs": [(P0, th0, phi1, L1), (joint, th0 + phi1, phi2, L2)]}


def biarc_cost(P0, th0, P1, th1) -> float | None:
    bi = biarc(P0, th0, P1, th1)
    return None if bi is None else bi["cost"]


def cocircularity_cost(P0, th0, P1, th1) -> float | None:
    """(a0 + a1)^2 with a0, a1 the angles of the two directions to the chord: 0 when both ends lie on one circle."""
    v = P1 - P0
    if v @ v < 1e-12:
        return None
    chord = np.arctan2(v[1], v[0])
    return float((wrap(th0 - chord) + wrap(th1 - chord)) ** 2)


GL_T, GL_W = np.polynomial.legendre.leggauss(48)
GL_T, GL_W = (GL_T + 1) / 2, GL_W / 2  # Gauss-Legendre points and weights on [0, 1]


def clothoid(P0, th0, P1, th1) -> dict | None:
    """The G1 clothoid (curvature changing linearly with length) from (P0, th0) to (P1, th1), after Bertolazzi and
    Frego, "G1 fitting with clothoids" (2015). Measured from the chord, the direction along the curve is
    phi0 + (delta - A) t + A t^2 for t in [0, 1]; Newton's method finds the A that ends the curve on the chord.
    Returns {"cost": delta^2 + A^2 / 3 (= length x integral of curvature^2), "L": length, "k0", "k1": curvature at
    the start and at the end, 1/px}."""
    v = P1 - P0
    r = float(np.hypot(*v))
    if r < 1e-9:
        return None
    phi = np.arctan2(v[1], v[0])
    phi0, phi1 = wrap(th0 - phi), wrap(th1 - phi)
    delta = phi1 - phi0
    A = 3.0 * (phi0 + phi1)
    for _ in range(50):
        ang = phi0 + (delta - A) * GL_T + A * GL_T ** 2
        y, dy = GL_W @ np.sin(ang), GL_W @ ((GL_T ** 2 - GL_T) * np.cos(ang))
        if abs(y) < 1e-13:
            break
        if abs(dy) < 1e-12:
            return None
        A -= float(np.clip(y / dy, -2.0, 2.0))
    ang = phi0 + (delta - A) * GL_T + A * GL_T ** 2
    x = float(GL_W @ np.cos(ang))
    if abs(GL_W @ np.sin(ang)) > 1e-9 or x <= 1e-9:
        return None
    L = r / x
    return {"cost": float(delta ** 2 + A ** 2 / 3), "L": L, "k0": float((delta - A) / L), "k1": float((delta + A) / L)}


def clothoid_cost(P0, th0, P1, th1) -> float | None:
    c = clothoid(P0, th0, P1, th1)
    return None if c is None else c["cost"]


COSTS = {"biarc": biarc_cost, "cocircularity": cocircularity_cost, "clothoid": clothoid_cost}


def sample_arc(P, th, phi, L, n: int = 60) -> np.ndarray:
    """n points along a circular arc from P with direction th, turning by phi over length L."""
    u = np.linspace(0, 1, n)
    if abs(phi) < 1e-9:
        return P + L * u[:, None] * np.array([np.cos(th), np.sin(th)])
    R, ang = L / phi, th + phi * u
    return np.c_[P[0] + R * (np.sin(ang) - np.sin(th)), P[1] - R * (np.cos(ang) - np.cos(th))]


def cuts_a_piece(bi: dict, cells: list[tuple[float, float, float, float]], slack: float) -> bool:
    """True when the biarc goes deeper than `slack` px into one of the cells (x0, x1, y0, y1)."""
    pts = np.vstack([sample_arc(*arc) for arc in bi["arcs"]])
    for x0, x1, y0, y1 in cells:
        depth = np.minimum.reduce([pts[:, 0] - x0, x1 - pts[:, 0], pts[:, 1] - y0, y1 - pts[:, 1]])
        if depth.max() > slack:
            return True
    return False
