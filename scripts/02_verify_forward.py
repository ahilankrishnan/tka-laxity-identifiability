"""Verification and baseline behavior of the forward simulator (nominal knee).

1. Contact-grid convergence of all 32 examination outputs and the time of one examination.
2. Neutral pose, ligament forces, and equilibrium residuals from 0 to 120 deg flexion (Figure 2A,B).
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.model import Knee, INSERT, GROUPS                   # noqa: E402
from knee.protocols import examine, neutral, _COMP            # noqa: E402

OUT = ROOT / "results" / "forward"


def main() -> None:
    """Run the grid-convergence study and the neutral flexion sweep."""
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    # 1. grid convergence, relative to the finest grid
    Y = {}
    for h in (3.0, 2.0, 1.5, 1.0, 0.5):
        t0 = time.time()
        Y[h], ok = examine(Knee(insert=dict(INSERT, h=h)))
        res[f"grid_{h}"] = dict(ok=bool(ok), time_s=time.time() - t0)
    # only grids whose examination converged have a meaningful error (a failed solve returns an arbitrary pose)
    res["grid_conv"] = {str(h): (float(np.abs(Y[h] - Y[0.5]).max()) if res[f"grid_{h}"]["ok"] else None)
                        for h in Y if h != 0.5}
    print("grid convergence (max abs error vs h = 0.5 mm):", res["grid_conv"])
    print("time of one examination at h = 1.5 mm: %.2f s" % res["grid_1.5"]["time_s"])

    # 2. neutral sweep
    k = Knee()
    flex = np.arange(0, 121, 5.0)
    Q, F, resid = [], [], []
    q, thp = None, None
    for f in flex:
        th = np.radians(f)
        q, _ = neutral(k, th, q, thp)
        thp = th
        resid.append(np.abs(k.generalized_forces(q, th, [_COMP])).max())
        R, t, _ = k.pose(q, th)
        _, _, fb, _ = k.ligament_forces(R, t)
        Q.append(q.copy())
        F.append(fb)
    res["max_neutral_residual"] = float(max(resid))
    print("max neutral residual:", res["max_neutral_residual"])

    np.savez(OUT / "forward.npz", flex=flex, Q=np.array(Q), F=np.array(F), names=k.names, group=k.group, groups=GROUPS)
    (OUT / "forward.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
