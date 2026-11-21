"""The full cost of joining two ends: geometry plus width, colour and curvature.

    C = C_join + w_width * q_width * C_width + w_colour * q_colour * C_colour + w_curvature * q_curvature * C_curvature

C_join       biarc, co-circularity or clothoid cost (geometry.py)
C_width      (width difference / mean width)^2
C_colour     (Lab colour difference / colour_scale)^2
C_curvature  ((curvature_a + curvature_b) * curvature_scale)^2: a curve that goes on bends the same way on both sides,
             and each end measures its bending looking outwards, so the two curvatures have opposite signs.
             With curvature_check = clothoid, each end is compared with the clothoid joining them instead:
             ((k0 - curvature_a) * scale)^2 + ((k1 + curvature_b) * scale)^2.
q            1 when both ends measured that feature reliably (extraction marks merged ends and poor fits), else 0,
             so an unreliable measurement adds nothing and only the geometry counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .geometry import COSTS, clothoid

if TYPE_CHECKING:
    from ..config import ScoreParams
    from .ends import End

SRGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def lab(rgb) -> np.ndarray:
    """sRGB (0-255) to CIE Lab (D65), so that colour differences match what the eye sees."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = SRGB_TO_XYZ @ c / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def join_cost(a: End, b: End, shift: np.ndarray, p: ScoreParams) -> float | None:
    """Cost of continuing end a (piece i) into end b (piece j placed at `shift` from piece i); None if impossible."""
    P0, th0, P1, th1 = a.pos, a.theta, b.pos + shift, b.theta + np.pi
    bridge = clothoid(P0, th0, P1, th1) if p.method == "clothoid" or (p.w_curvature and p.curvature_check == "clothoid") else None
    c = (bridge or {}).get("cost") if p.method == "clothoid" else COSTS[p.method](P0, th0, P1, th1)
    if c is None:
        return None
    if p.w_width and a.width_ok and b.width_ok:
        c += p.w_width * (2 * (a.width - b.width) / (a.width + b.width)) ** 2
    if p.w_colour and a.lab is not None and b.lab is not None:
        c += p.w_colour * (np.linalg.norm(a.lab - b.lab) / p.colour_scale) ** 2
    if p.w_curvature and a.curvature is not None and b.curvature is not None:
        if p.curvature_check == "ends":
            c += p.w_curvature * ((a.curvature + b.curvature) * p.curvature_scale) ** 2
        elif bridge is not None:
            c += p.w_curvature * (((bridge["k0"] - a.curvature) * p.curvature_scale) ** 2
                                  + ((bridge["k1"] + b.curvature) * p.curvature_scale) ** 2)
    return float(c)
