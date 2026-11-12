"""Rigid-body tibiofemoral model of a cruciate-retaining total knee arthroplasty.

The model represents a right knee.  Femoral frame (mm): x anterior, y medial, z proximal; the
origin lies midway between the condyles on the femoral flexion axis, which coincides with the y axis.
Tibial frame: same axis directions in the reference pose, with the origin midway between the lowest
points of the medial and lateral insert dishes, 25 mm below the femoral origin at full extension.
The tibia is a rigid body whose pose relative to the femur is

    p_F = R(q) p_T + Ry(theta) u,      R = Ry(theta) Rx(varus) Rz(internal),

with prescribed flexion theta and five unknown coordinates
q = [anterior, medial, proximal translation (mm), varus, internal rotation (rad)], where u is the
vector from the femoral to the tibial origin in the flexed femoral frame (u = [0, 0, -25] in the
reference pose).

Articular contact uses an elastic-foundation (penalty) law between analytic toroidal
femoral condyles and paraboloidal tibial insert dishes; ligaments are tension-only
nonlinear springs (Blankevoort and Huiskes, 1991).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

GROUPS = ("MCL", "LCL", "PCL", "CAP")
EPS_L = 0.03          # transition strain of the toe region (Blankevoort and Huiskes, 1991)
TOL = 1e-2            # accepted residual of an equilibrium solve, N (moments scaled by MOMENT_SCALE)
NEWTON_TOL = 1e-4     # residual at which Newton iterations stop, N
FD_STEP = 1e-6        # forward-difference step for the Jacobian, mm or rad
MOMENT_SCALE = 25.0   # mm; divides moments so that force and moment residuals have similar size
SMOOTH_SQ = 1e-4      # mm^2; squared width of the smoothed positive part in the contact law (0.01 mm)
STEP_MAX = np.array([2.0, 2.0, 1.0, 0.05, 0.1])  # Newton step limits (mm, mm, mm, rad, rad)
# Seating at full extension and the gap (distraction) solves apply their load in INCREMENTS[0] equal
# increments; if any increment fails to converge, the whole solve is repeated with the next, finer schedule.
INCREMENTS = (5, 10, 20, 40)
# Flexion of the seated knee and every test load use adaptive continuation (see Knee._continue): an increment
# is accepted only if it converges and moves no coordinate by more than PATH_DQ, otherwise it is halved.  The
# solution thus follows one equilibrium branch continuously, independent of how far a Newton step could jump.
PATH_DQ = np.array([1.0, 1.0, 1.0, 0.03, 0.03])   # mm, mm, mm, rad, rad per accepted increment
PATH_STEP0 = 0.2          # initial and largest load-factor increment
PATH_STEP_MIN = 1.0 / 4096   # smallest increment; below it the branch counts as not followable

# Load = list of (force, point of application, moment) triples, each a 3-vector in the tibial frame.
Load = Optional[Sequence[tuple]]


def Rx(a: float) -> np.ndarray:
    """Rotation matrix about the x axis by angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a: float) -> np.ndarray:
    """Rotation matrix about the y axis (flexion axis) by angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a: float) -> np.ndarray:
    """Rotation matrix about the z axis by angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------- geometry
# Femoral component: two tori about the flexion axis (y), one per condyle.
FEM = dict(R_sag=25.0, r_cor=20.0, y_med=22.0, y_lat=-22.0)
# Tibial insert dishes in the tibial frame: z = z0 + (x-xc)^2/(2 R_ap) + (y-yc)^2/(2 R_ml)
INSERT = dict(z0=0.0, xc=0.0, R_ap_med=45.0, R_ap_lat=80.0, R_ml=26.0,
              x_min=-28.0, x_max=22.0, half_width=13.5, h=1.5)

# Position of the tibial origin in the femoral frame in the reference pose (full extension), mm.
T_REF = np.array([0.0, 0.0, -25.0])
Q_REF = np.r_[T_REF, 0.0, 0.0]

# Ligament bundles: femoral origin (femoral frame) and tibial insertion (tibial frame), mm.
BUNDLES = {
    # name      group   femoral origin            tibial insertion
    "aMCL": ("MCL", (4.0, 40.0, 4.0), (12.0, 32.0, -60.0)),
    "iMCL": ("MCL", (0.0, 41.0, 4.0), (2.0, 33.0, -57.0)),
    "pMCL": ("MCL", (-6.0, 40.0, 8.0), (-20.0, 33.0, -20.0)),
    "LCL":  ("LCL", (-3.0, -42.0, 4.0), (-24.0, -38.0, -38.0)),
    "aPCL": ("PCL", (4.0, 9.0, -3.0), (-24.0, 2.0, -11.0)),
    "pPCL": ("PCL", (-4.0, 13.0, 0.0), (-27.0, 5.0, -15.0)),
    "mCAP": ("CAP", (-28.0, 24.0, 14.0), (-27.0, 24.0, -7.0)),
    "lCAP": ("CAP", (-28.0, -24.0, 14.0), (-27.0, -24.0, -7.0)),
    "alCAP": ("CAP", (-5.0, -41.0, 6.0), (-2.0, -35.0, -6.0)),
}

# Bundles with the tibial insertion expressed in the femoral frame of the reference pose (for figures/tables).
BUNDLES_REF = {n: (g, pf, tuple(np.add(pt, T_REF))) for n, (g, pf, pt) in BUNDLES.items()}

# Nominal ligament parameters (stiffness N per unit strain, reference strain), inside the prior ranges of
# sampling.PRIOR; used for the forward checks (02_verify_forward.py) and as the default Knee.
NOMINAL = dict(k_MCL=2750.0, eps_MCL=0.04, k_LCL=2000.0, eps_LCL=0.04,
               k_PCL=4500.0, eps_PCL=0.0, k_CAP=1500.0, eps_CAP=0.0)


def _insert_grid(ins: dict) -> tuple[np.ndarray, np.ndarray]:
    """Contact sample points on both insert dishes (tibial frame) and their compartment (0 medial, 1 lateral)."""
    xs = np.arange(ins["x_min"], ins["x_max"] + 1e-9, ins["h"])
    ws = np.arange(-ins["half_width"], ins["half_width"] + 1e-9, ins["h"])
    X, W = np.meshgrid(xs, ws, indexing="ij")
    X, W = X.ravel(), W.ravel()
    pts, comp = [], []
    for yc, Rap, c in ((FEM["y_med"], ins["R_ap_med"], 0), (FEM["y_lat"], ins["R_ap_lat"], 1)):
        Y = yc + W
        Z = ins["z0"] + (X - ins["xc"]) ** 2 / (2 * Rap) + W ** 2 / (2 * ins["R_ml"])
        pts.append(np.c_[X, Y, Z])
        comp.append(np.full(X.size, c))
    return np.vstack(pts), np.concatenate(comp)


@dataclass
class Knee:
    """One virtual knee: ligament parameters plus (optionally perturbed) geometry."""
    params: dict = field(default_factory=lambda: dict(NOMINAL))
    k_contact: float = 250.0                      # elastic-foundation modulus, N/mm^3
    attach_offsets: Optional[np.ndarray] = None   # (n_bundles, 2, 3) perturbations, mm
    insert: dict = field(default_factory=lambda: dict(INSERT))

    def __post_init__(self) -> None:
        """Assemble attachment sites, contact grid, and reference lengths."""
        names = list(BUNDLES)
        self.names = names
        self.group = np.array([GROUPS.index(BUNDLES[n][0]) for n in names])
        pf = np.array([BUNDLES[n][1] for n in names], float)
        pt = np.array([BUNDLES[n][2] for n in names], float)
        if self.attach_offsets is not None:
            pf = pf + self.attach_offsets[:, 0]
            pt = pt + self.attach_offsets[:, 1]
        self.pf, self.pt = pf, pt
        self.grid, self.comp = _insert_grid(self.insert)
        self.area = self.insert["h"] ** 2
        self.yc = np.where(self.comp == 0, FEM["y_med"], FEM["y_lat"])
        self.a_tor = FEM["R_sag"] - FEM["r_cor"]
        self.set_params(self.params)

    def set_params(self, params: dict) -> None:
        """Set group stiffness k and reference strain eps; slack length L0 = L_ref / (1 + eps)."""
        self.params = dict(params)
        k = np.array([params[f"k_{g}"] for g in GROUPS])
        e = np.array([params[f"eps_{g}"] for g in GROUPS])
        self.k = k[self.group]
        L_ref = np.linalg.norm(self.pf - (self.pt + T_REF), axis=1)  # reference pose: extension, q = Q_REF
        self.L0 = L_ref / (1.0 + e[self.group])

    # ------------------------------------------------------------ kinematics
    @staticmethod
    def pose(q: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tibial rotation R, translation t, and flexion rotation Rf for coordinates q at flexion theta."""
        Rf = Ry(theta)
        R = Rf @ Rx(q[3]) @ Rz(q[4])
        t = Rf @ q[:3]
        return R, t, Rf

    # ------------------------------------------------------------ elements
    def ligament_forces(self, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tibial insertion points, force vectors on the tibia, force magnitudes, and strains of all bundles."""
        P = self.pt @ R.T + t
        d = self.pf - P
        L = np.linalg.norm(d, axis=1)
        eps = L / self.L0 - 1.0
        f = np.where(eps <= 0, 0.0,
                     np.where(eps < 2 * EPS_L, 0.25 * self.k * eps ** 2 / EPS_L,
                              self.k * (eps - EPS_L)))
        return P, (f / L)[:, None] * d, f, eps

    def signed_distance(self, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed distance of points G (femoral frame) to the toroidal condyles and the outward normals."""
        rho = np.hypot(G[:, 0], G[:, 2])
        dy = G[:, 1] - self.yc
        dr = rho - self.a_tor
        s = np.hypot(dr, dy)
        dist = s - FEM["r_cor"]
        n = np.c_[dr / s * G[:, 0] / rho, dy / s, dr / s * G[:, 2] / rho]
        return dist, n

    def contact_forces(self, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Elastic-foundation contact: grid points, force vectors on the tibia, and signed distances."""
        G = self.grid @ R.T + t
        dist, n = self.signed_distance(G)
        pen = 0.5 * (-dist + np.sqrt(dist ** 2 + SMOOTH_SQ))   # smoothed positive part of -d
        F = (self.k_contact * self.area * pen)[:, None] * n
        return G, F, dist

    # ------------------------------------------------------------ equilibrium
    def generalized_forces(self, q: np.ndarray, theta: float, load: Load) -> np.ndarray:
        """Generalized forces along the five coordinates (moments divided by MOMENT_SCALE)."""
        R, t, Rf = self.pose(q, theta)
        P, Fl, _, _ = self.ligament_forces(R, t)
        G, Fc, _ = self.contact_forces(R, t)
        F = Fl.sum(0) + Fc.sum(0)
        M = np.cross(P, Fl).sum(0) + np.cross(G, Fc).sum(0)
        if load is not None:
            for force_T, point_T, moment_T in load:
                Fe = R @ np.asarray(force_T, float)
                F = F + Fe
                M = M + np.cross(R @ np.asarray(point_T, float) + t, Fe) + R @ np.asarray(moment_T, float)
        M = M - np.cross(t, F)                     # moment about the tibial origin
        ax_ie = Rf @ Rx(q[3]) @ np.array([0.0, 0.0, 1.0])
        return np.array([F @ Rf[:, 0], F @ Rf[:, 1], F @ Rf[:, 2],
                         (M @ Rf[:, 0]) / MOMENT_SCALE, (M @ ax_ie) / MOMENT_SCALE])

    def solve(self, theta: float, load: Load = None, q0: Optional[np.ndarray] = None,
              fixed: Optional[Sequence[int]] = None, steps: int = 1) -> tuple[np.ndarray, bool]:
        """Static equilibrium at flexion theta (rad) with the load applied in `steps` increments.

        `fixed` lists coordinate indices held at their values in q0."""
        q = Q_REF.copy() if q0 is None else np.array(q0, float)
        free = np.array([i for i in range(5) if not fixed or i not in fixed])
        ok = True
        for s in range(1, steps + 1):
            lam = s / steps
            ld = None if load is None else [(lam * np.asarray(f), p, lam * np.asarray(m)) for f, p, m in load]

            def res(x: np.ndarray) -> np.ndarray:
                """Residual of the free coordinates, with the fixed ones held at q."""
                qq = q.copy()
                qq[free] = x
                return self.generalized_forces(qq, theta, ld)[free]

            x, r = self._newton(res, q[free], STEP_MAX[free])
            q[free] = x
            ok &= bool(np.all(np.isfinite(r)) and np.max(np.abs(r)) < TOL)   # every increment must converge
        return q, ok

    def solve_robust(self, theta: float, load: Load = None, q0: Optional[np.ndarray] = None,
                     fixed: Optional[Sequence[int]] = None) -> tuple[np.ndarray, bool]:
        """Solve along the fixed loading path (INCREMENTS[0] increments), refining it until every increment converges."""
        for steps in INCREMENTS:
            q, ok = self.solve(theta, load, q0, fixed, steps)
            if ok:
                break
        return q, ok

    def _continue(self, residual: Callable[[float], Callable[[np.ndarray], np.ndarray]], q0: np.ndarray,
                  dq_max: Optional[np.ndarray] = None) -> tuple[np.ndarray, bool]:
        """Follow an equilibrium branch from q0 (the solution at lambda = 0) to lambda = 1.

        residual(lam) returns the residual function at continuation parameter lam.  An increment is accepted
        only if Newton converges and no coordinate moves by more than dq_max (default PATH_DQ); otherwise the
        increment is halved.  Returns the end point and whether the whole branch was followed."""
        dq_max = PATH_DQ if dq_max is None else dq_max
        q = np.array(q0, float)
        lam, dlam = 0.0, PATH_STEP0
        while lam < 1.0 - 1e-12:
            dl = min(dlam, 1.0 - lam)
            lt = lam + dl
            x, r = self._newton(residual(lt), q, STEP_MAX)
            conv = bool(np.all(np.isfinite(r)) and np.max(np.abs(r)) < TOL)
            if conv and np.all(np.abs(x - q) <= dq_max):
                q, lam = x, lt
                dlam = min(2.0 * dl, PATH_STEP0)
            elif dl > PATH_STEP_MIN:
                dlam = dl / 2.0
            else:
                return q, False      # the branch cannot be followed continuously (limit point or no convergence)
        return q, True

    def solve_path(self, theta: float, load: Load, q0: np.ndarray, base: Load = None,
                   dq_max: Optional[np.ndarray] = None) -> tuple[np.ndarray, bool]:
        """Equilibrium under `base` + `load`, following the branch from q0, the equilibrium under `base` alone.

        The load factor of `load` grows from 0 to 1 by adaptive continuation (see `_continue`); the end state
        must also be statically stable."""
        base = list(base or [])

        def residual(lt: float) -> Callable[[np.ndarray], np.ndarray]:
            """Residual function at load factor lt."""
            ld = base + [(lt * np.asarray(f), p, lt * np.asarray(m)) for f, p, m in load]
            return lambda x: self.generalized_forces(x, theta, ld)

        q, ok = self._continue(residual, q0, dq_max)
        return q, ok and self.is_stable(q, theta, base + list(load))

    def solve_flex_path(self, theta0: float, theta1: float, load: Load, q0: np.ndarray,
                        dq_max: Optional[np.ndarray] = None) -> tuple[np.ndarray, bool]:
        """Equilibrium under `load` at flexion theta1, following the branch from q0, the equilibrium at theta0.

        Flexion moves from theta0 to theta1 by adaptive continuation (see `_continue`); the end state must
        also be statically stable."""
        def residual(lt: float) -> Callable[[np.ndarray], np.ndarray]:
            """Residual function at flexion theta0 + lt (theta1 - theta0)."""
            th = theta0 + lt * (theta1 - theta0)
            return lambda x: self.generalized_forces(x, th, load)

        q, ok = self._continue(residual, q0, dq_max)
        return q, ok and self.is_stable(q, theta1, load)

    def is_stable(self, q: np.ndarray, theta: float, load: Load) -> bool:
        """Static stability of an equilibrium: every eigenvalue of dQ/dq has a negative real part."""
        res = lambda x: self.generalized_forces(x, theta, load)
        J = self._fd_jacobian(res, q, res(q))
        return bool(np.max(np.linalg.eigvals(J).real) < 0.0)

    @staticmethod
    def _fd_jacobian(res: Callable[[np.ndarray], np.ndarray], x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Forward-difference Jacobian of res at x, where r = res(x)."""
        n = x.size
        J = np.empty((r.size, n))
        for j in range(n):
            xp = x.copy()
            xp[j] += FD_STEP
            J[:, j] = (res(xp) - r) / FD_STEP
        return J

    @staticmethod
    def _newton(res: Callable[[np.ndarray], np.ndarray], x: np.ndarray, smax: np.ndarray,
                maxit: int = 60) -> tuple[np.ndarray, np.ndarray]:
        """Newton iteration with forward-difference Jacobian and per-coordinate step limits."""
        x = np.array(x, float)
        r = res(x)
        for _ in range(maxit):
            if np.max(np.abs(r)) < NEWTON_TOL:
                break
            J = Knee._fd_jacobian(res, x, r)
            try:
                dx = -np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                dx = -np.linalg.lstsq(J, r, rcond=None)[0]
            x = x + dx / max(1.0, np.max(np.abs(dx) / smax))
            r = res(x)
        return x, r

    def gaps(self, q: np.ndarray, theta: float) -> np.ndarray:
        """Minimum condyle-insert separation in the medial and lateral compartments, mm."""
        R, t, _ = self.pose(q, theta)
        _, _, dist = self.contact_forces(R, t)
        return np.array([dist[self.comp == 0].min(), dist[self.comp == 1].min()])
