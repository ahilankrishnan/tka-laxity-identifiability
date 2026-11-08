"""Surface mesh of the virtual knee in the reference pose (full extension, q = Q_REF) for the figures.

The femoral condyles and insert dishes follow the analytic contact geometry of `model.py`; the bones are
schematic lofts sized to an adult knee and serve only for visualization. Coordinates are in millimeters in the
femoral frame of a right knee (x anterior, y medial, z proximal).
"""
from __future__ import annotations

import numpy as np
from .model import FEM, INSERT, BUNDLES_REF, T_REF

COLORS = {  # RGB in [0, 1]
    "femur": (0.89, 0.84, 0.72), "tibia": (0.89, 0.84, 0.72), "fibula": (0.89, 0.84, 0.72),
    "femoral_component": (0.72, 0.74, 0.78), "insert": (0.96, 0.95, 0.90),
    "MCL": (0.122, 0.471, 0.706), "LCL": (0.698, 0.094, 0.169), "PCL": (0.106, 0.620, 0.467),
    "CAP": (0.416, 0.239, 0.604),
}


def _grid(P: np.ndarray, closed_u: bool = False) -> np.ndarray:
    """Triangles of a structured (nu, nv, 3) point grid, optionally closed in u."""
    nu, nv = P.shape[:2]
    idx = np.arange(nu * nv).reshape(nu, nv)
    F = []
    for i in range(nu if closed_u else nu - 1):
        i2 = (i + 1) % nu
        for j in range(nv - 1):
            a, b, c, d = idx[i, j], idx[i2, j], idx[i2, j + 1], idx[i, j + 1]
            F += [(a, b, c), (a, c, d)]
    return np.array(F)


def _loft(sections: list, n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Closed loft through elliptical sections (z, x_center, y_center, half-width x, half-width y)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.array([[(xc + ax * np.cos(tt), yc + ay * np.sin(tt), z) for (z, xc, yc, ax, ay) in sections] for tt in t])
    return P.reshape(-1, 3), _grid(P, closed_u=True)


def _ellipsoid(c, r, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.linspace(0.05, np.pi - 0.05, n // 2)
    P = np.array([[(c[0] + r[0] * np.cos(uu) * np.sin(vv), c[1] + r[1] * np.sin(uu) * np.sin(vv), c[2] + r[2] * np.cos(vv))
                   for vv in v] for uu in u])
    return P.reshape(-1, 3), _grid(P, closed_u=True)


def _merge(parts) -> tuple[np.ndarray, np.ndarray]:
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + off); off += len(v)
    return np.vstack(V), np.vstack(F)


def _tube(p0, p1, r: float, n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; d /= np.linalg.norm(d)
    a = np.cross(d, [1.0, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1.0, 0])
    a /= np.linalg.norm(a); b = np.cross(d, a)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.cos(t)[:, None] * a + np.sin(t)[:, None] * b
    P = np.stack([p0 + r * ring, p1 + r * ring], axis=1)
    return P.reshape(-1, 3), _grid(P, closed_u=True)


def build() -> tuple[dict, list]:
    """Parts {name: (vertices, triangles)} and ligament segments [(bundle, group, femoral, tibial)]."""
    parts = {}
    # femoral component: toroidal condyles (sagittal radius R_sag, coronal radius r_cor)
    a, r = FEM["R_sag"] - FEM["r_cor"], FEM["r_cor"]
    # outer sheet only: with a < r the torus is a spindle torus whose inner sheet is not a contact surface
    pm = np.arccos(-a / r)
    # articular arc from posterior through distal to anterior, closed by planar sagittal caps at both ends
    th = np.radians(np.linspace(150, 382, 70)); ph = np.linspace(-pm, pm, 41)
    cond = []
    for yc in (FEM["y_med"], FEM["y_lat"]):
        P = np.array([[((a + r * np.cos(p)) * np.cos(t), yc + r * np.sin(p), (a + r * np.cos(p)) * np.sin(t)) for p in ph] for t in th])
        V, F = P.reshape(-1, 3), _grid(P)
        caps = []
        for row in (P[0], P[-1]):
            c0 = len(V) + len(caps) * (len(ph) + 1)
            caps.append(np.vstack([[0.0, yc, 0.0], row]))
            F = np.vstack([F, [(c0, c0 + 1 + j, c0 + 2 + j) for j in range(len(ph) - 1)]])
        cond.append((np.vstack([V] + caps), F))
    parts["femoral_component"] = _merge(cond)
    # femur: distal metaphysis flaring into the shaft
    parts["femur"] = _merge([_loft([(8, -4, 0, 27, 41), (25, -2, 0, 24, 36), (45, 0, 0, 18, 24), (70, 2, 0, 14, 15),
                                     (95, 3, 0, 13, 13)])])
    # insert: dished top surface over the tray footprint, flat bottom, side walls
    ins = INSERT
    xs = np.linspace(ins["x_min"], ins["x_max"], 34); ys = np.linspace(-36, 36, 49)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.full_like(X, -19.0)
    for yc, Rap in ((FEM["y_med"], ins["R_ap_med"]), (FEM["y_lat"], ins["R_ap_lat"])):
        Z = np.minimum(Z, ins["z0"] + T_REF[2] + (X - ins["xc"]) ** 2 / (2 * Rap) + (Y - yc) ** 2 / (2 * ins["R_ml"]))
    top = np.stack([X, Y, Z], -1)
    ring = np.vstack([top[:, 0], top[-1, 1:], top[::-1, -1][1:], top[0, ::-1][1:-1]])
    wall = np.stack([ring, ring * [1, 1, 0] + [0, 0, -35.0]], axis=1)
    bot = top.copy(); bot[..., 2] = -35.0
    parts["insert"] = _merge([(top.reshape(-1, 3), _grid(top)), (wall.reshape(-1, 3), _grid(wall, closed_u=True)),
                              (bot.reshape(-1, 3), _grid(bot)[:, ::-1])])
    # tibia and fibula
    parts["tibia"] = _merge([_loft([(-35.5, -3, 0, 27, 38), (-45, -4, 0, 26, 36), (-60, -5, 0, 21, 27), (-80, -4, 2, 16, 18),
                                    (-96, -3, 3, 14, 14)])])
    parts["fibula"] = _merge([_ellipsoid((-27, -43, -70), (8, 8, 9)), _loft([(-72, -28, -45, 5.5, 5.5), (-96, -29, -46, 5, 5)], 16)])
    ligs = [(n, g, np.array(pf), np.array(pt)) for n, (g, pf, pt) in BUNDLES_REF.items()]
    for g in ("MCL", "LCL", "PCL", "CAP"):
        rad = {"MCL": 2.2, "LCL": 2.4, "PCL": 2.8, "CAP": 1.8}[g]
        parts[g] = _merge([_tube(pf, pt, rad) for _, gg, pf, pt in ligs if gg == g])
    return parts, ligs
