"""Surface meshes of the implant in the reference pose (full extension, q = Q_REF) for Fig. 1.

The femoral condyles and insert dishes follow the analytic contact geometry of `model.py`.  Coordinates are in
millimeters in the femoral frame of a right knee (x anterior, y medial, z proximal).
"""
from __future__ import annotations

import numpy as np
from .model import FEM, INSERT, T_REF

COLORS = {"femoral_component": (0.72, 0.74, 0.78), "insert": (0.96, 0.95, 0.90)}   # RGB in [0, 1]
INSERT_HALF_WIDTH = 36.0    # mm, medial-lateral half width of the drawn insert
INSERT_BOTTOM_Z = -35.0     # mm, femoral-frame height of the drawn insert's flat bottom
INSERT_RIM_Z = -19.0        # mm, highest drawn point of the dished top surface (rim)


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


def _merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """One mesh (vertices, triangles) from several, with triangle indices offset accordingly."""
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + off); off += len(v)
    return np.vstack(V), np.vstack(F)


def build() -> dict:
    """Implant parts {name: (vertices, triangles)}: femoral component and tibial insert."""
    parts = {}
    # femoral component: toroidal condyles (sagittal radius R_sag, coronal radius r_cor)
    a, r = FEM["R_sag"] - FEM["r_cor"], FEM["r_cor"]
    # outer sheet only: with a < r the torus is a spindle torus whose inner sheet is not a contact surface
    pm = np.arccos(-a / r)
    # articular arc from posterior through distal to anterior, closed by planar sagittal caps at both ends
    th = np.radians(np.linspace(150, 382, 70)); ph = np.linspace(-pm, pm, 41)
    cond = []
    for yc in (FEM["y_med"], FEM["y_lat"]):
        P = np.array([[((a + r * np.cos(p)) * np.cos(t), yc + r * np.sin(p), (a + r * np.cos(p)) * np.sin(t))
                       for p in ph] for t in th])
        V, F = P.reshape(-1, 3), _grid(P)
        caps = []
        for row in (P[0], P[-1]):
            c0 = len(V) + len(caps) * (len(ph) + 1)
            caps.append(np.vstack([[0.0, yc, 0.0], row]))
            F = np.vstack([F, [(c0, c0 + 1 + j, c0 + 2 + j) for j in range(len(ph) - 1)]])
        cond.append((np.vstack([V] + caps), F))
    parts["femoral_component"] = _merge(cond)
    # insert: dished top surface over the tray footprint, flat bottom, side walls
    ins = INSERT
    xs = np.linspace(ins["x_min"], ins["x_max"], 34); ys = np.linspace(-INSERT_HALF_WIDTH, INSERT_HALF_WIDTH, 49)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.full_like(X, INSERT_RIM_Z)
    for yc, Rap in ((FEM["y_med"], ins["R_ap_med"]), (FEM["y_lat"], ins["R_ap_lat"])):
        Z = np.minimum(Z, ins["z0"] + T_REF[2] + (X - ins["xc"]) ** 2 / (2 * Rap) + (Y - yc) ** 2 / (2 * ins["R_ml"]))
    top = np.stack([X, Y, Z], -1)
    ring = np.vstack([top[:, 0], top[-1, 1:], top[::-1, -1][1:], top[0, ::-1][1:-1]])
    wall = np.stack([ring, ring * [1, 1, 0] + [0, 0, INSERT_BOTTOM_Z]], axis=1)
    bot = top.copy(); bot[..., 2] = INSERT_BOTTOM_Z
    parts["insert"] = _merge([(top.reshape(-1, 3), _grid(top)), (wall.reshape(-1, 3), _grid(wall, closed_u=True)),
                              (bot.reshape(-1, 3), _grid(bot)[:, ::-1])])
    return parts
