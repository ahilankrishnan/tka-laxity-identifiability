"""Virtual laxity examinations: AP drawer, varus-valgus, internal-external rotation, joint gaps."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from .model import Knee, GROUPS

FLEX_TRAIN = (0.0, 30.0, 60.0, 90.0)      # flexion angles used for inference, deg
FLEX_HELDOUT = (15.0, 45.0, 75.0)         # flexion angles withheld from inference, deg
COMPRESSION = 50.0     # N, small axial load that seats the femur in the insert
AP_FORCE = 100.0       # N, applied 5 mm below the joint line (AP_ARM_Z)
VV_MOMENT = 10.0e3     # N mm
IE_MOMENT = 3.0e3      # N mm
DISTRACTION = 100.0    # N per compartment (spreader)

JOINT_LINE_Z = 0.0    # mm, tibial-frame height of the lowest insert points (joint line, tibial origin)
AP_ARM_Z = -5.0       # mm, height at which the AP force is applied

_COMP = ([0.0, 0.0, COMPRESSION], [0.0, 0.0, JOINT_LINE_Z], [0.0, 0.0, 0.0])
_TESTS = {  # name: (load list in tibial frame, coordinate index, scale to report)
    "ant": ([([AP_FORCE, 0, 0], [0, 0, AP_ARM_Z], [0, 0, 0])], 0, 1.0),
    "post": ([([-AP_FORCE, 0, 0], [0, 0, AP_ARM_Z], [0, 0, 0])], 0, 1.0),
    "var": ([([0, 0, 0], [0, 0, 0], [VV_MOMENT, 0, 0])], 3, 180 / np.pi),
    "val": ([([0, 0, 0], [0, 0, 0], [-VV_MOMENT, 0, 0])], 3, 180 / np.pi),
    "int": ([([0, 0, 0], [0, 0, 0], [0, 0, IE_MOMENT])], 4, 180 / np.pi),
    "ext": ([([0, 0, 0], [0, 0, 0], [0, 0, -IE_MOMENT])], 4, 180 / np.pi),
}
TEST_UNITS = {"ant": "mm", "post": "mm", "var": "deg", "val": "deg", "int": "deg", "ext": "deg",
              "gapM": "mm", "gapL": "mm"}
COMBINED = {  # held-out combined loads at 45 deg flexion: (load list, reported coordinates)
    "var+int": [([0, 0, 0], [0, 0, 0], [VV_MOMENT, 0, IE_MOMENT])],
    "val+ext": [([0, 0, 0], [0, 0, 0], [-VV_MOMENT, 0, -IE_MOMENT])],
    "ant+int": [([AP_FORCE, 0, 0], [0, 0, AP_ARM_Z], [0, 0, IE_MOMENT])],
}


def feature_names(flex: Sequence[float] = FLEX_TRAIN,
                  tests: Sequence[str] = ("ant", "post", "var", "val", "int", "ext", "gapM", "gapL")) -> list[str]:
    """Names of the examination outputs, ordered as returned by `examine`."""
    return [f"{t}@{int(f)}" for f in flex for t in tests]


def neutral(knee: Knee, theta: float, q0: Optional[np.ndarray] = None,
            theta0: Optional[float] = None) -> tuple[np.ndarray, bool]:
    """Equilibrium pose under the seating compression alone (the seated pose).

    The knee is seated at full extension from the reference pose and then flexed to theta by adaptive
    continuation, so the seated pose at every angle lies on one continuous branch.  When q0 is the seated pose
    at flexion theta0 (rad), the continuation starts from there."""
    if q0 is None:
        q0, ok = knee.solve_robust(0.0, [_COMP])
        ok = ok and knee.is_stable(q0, 0.0, [_COMP])
        theta0 = 0.0
    else:
        ok = True
    if theta != theta0:
        q0, ok2 = knee.solve_flex_path(theta0, theta, [_COMP], q0)
        ok = ok and ok2
    return q0, ok


def laxity_at(knee: Knee, flex_deg: float, q_prev: Optional[np.ndarray] = None, with_gaps: bool = True,
              extra: Optional[dict] = None, flex_prev: Optional[float] = None) -> tuple[dict, np.ndarray, bool]:
    """Laxity outputs at one flexion angle; returns dict, seated pose, convergence flag.

    q_prev is the seated pose at flexion flex_prev (deg), from which the knee is flexed (None: from extension)."""
    th = np.radians(flex_deg)
    q0, ok = neutral(knee, th, q_prev, None if q_prev is None else np.radians(flex_prev))
    out, allok = {}, ok
    for name, (load, idx, sc) in _TESTS.items():
        q, ok = knee.solve_path(th, load, q0, base=[_COMP])
        out[name] = (q[idx] - q0[idx]) * sc
        allok &= ok
    if with_gaps:
        dist = [([0, 0, -DISTRACTION], [0, FEMY, JOINT_LINE_Z], [0, 0, 0]) for FEMY in (-22.0, 22.0)]
        qs = q0.copy()
        qs[2] -= 0.5
        q, ok = knee.solve_robust(th, dist, q0=qs, fixed=[0, 1, 4])
        out["gapM"], out["gapL"] = knee.gaps(q, th)
        allok &= ok
    if extra:
        for name, load in extra.items():
            q, ok = knee.solve_path(th, load, q0, base=[_COMP])
            out[name + ":ap"] = q[0] - q0[0]
            out[name + ":vv"] = np.degrees(q[3] - q0[3])
            out[name + ":ie"] = np.degrees(q[4] - q0[4])
            allok &= ok
    return out, q0, allok


def examine(knee: Knee, flex: Sequence[float] = FLEX_TRAIN, tests: Optional[Sequence[str]] = None,
            with_gaps: bool = True) -> tuple[np.ndarray, bool]:
    """Full examination over flexion; returns feature vector (ordered by feature_names) and flag."""
    tests = tests or ("ant", "post", "var", "val", "int", "ext", "gapM", "gapL")
    vals, allok, q, fp = [], True, None, None
    for f in flex:
        o, q, ok = laxity_at(knee, f, q, with_gaps=with_gaps, flex_prev=fp)
        fp = f
        allok &= ok
        vals += [o[t] for t in tests]
    return np.array(vals), allok


def heldout(knee: Knee) -> tuple[list[str], np.ndarray, bool]:
    """Outputs under loading conditions excluded from inference."""
    names, vals, allok, q, fp = [], [], True, None, None
    for f in FLEX_HELDOUT:
        o, q, ok = laxity_at(knee, f, q, with_gaps=False, extra=COMBINED if f == 45.0 else None, flex_prev=fp)
        fp = f
        allok &= ok
        for k_, v in o.items():
            names.append(f"{k_}@{int(f)}")
            vals.append(v)
    return names, np.array(vals), allok
