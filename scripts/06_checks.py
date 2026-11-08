"""Numerical checks reported in the Supplementary Material.

1. Path dependence: examinations of test knees 40-119 with the default solver versus every load applied
   in 10 increments; counts outputs that differ by more than 0.05 mm or deg.
2. Sensitivity outliers: test knees whose noise-normalized sensitivities exceed 300 (a central difference
   spanning two equilibria) and the largest change of the median identifiability indices without them.
3. Step check: largest change of the median indices between steps 0.02 and 0.001 on the first 50 knees.
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import knee.model as M                                        # noqa: E402
from knee.sampling import from_vector                         # noqa: E402
from knee.protocols import examine                            # noqa: E402
from knee.subsets import SUBSETS, subset_index, noise_sd      # noqa: E402

z = np.load(ROOT / "results" / "data" / "test_nominal.npz")
D = {k: z[k] for k in ("theta", "offsets")}
IDF = ROOT / "results" / "identifiability"


def incremental(self, theta, load=None, q0=None, fixed=None):
    for steps in (10, 20):
        q, ok = self.solve(theta, load, q0, fixed, steps)
        if ok:
            break
    return q, ok


def job(i):
    k = M.Knee(params=from_vector(D["theta"][i]), attach_offsets=D["offsets"][i])
    default = M.Knee.solve_robust
    y1, _ = examine(k)
    M.Knee.solve_robust = incremental
    try:
        y10, _ = examine(k)
    finally:
        M.Knee.solve_robust = default
    return np.abs(y1 - y10)


def medians(Js, name):
    idx = subset_index(name); s = noise_sd()[idx]; out = []
    for J in Js:
        a = J[idx] / s[:, None]
        C = np.linalg.inv(a.T @ a + 12 * np.eye(8))
        out.append(1 - np.sqrt(np.diag(C)) / np.sqrt(1 / 12))
    return np.median(out, 0)


if __name__ == "__main__":
    with Pool(4) as p:
        d = np.array(p.map(job, range(40, 120)))
    out = dict(path_knees=80, path_outputs=int(d.size), path_changed=int((d > 0.05).sum()))
    J = np.load(IDF / "summary_J.npy")
    m = (np.abs(J) / noise_sd()[None, :, None]).max(axis=(1, 2))
    bad = np.where(m > 300)[0]; good = np.setdiff1d(np.arange(len(J)), bad)
    out.update(fd_outlier_knees=int(bad.size),
               fd_outlier_max_change=float(max(np.abs(medians(J, n) - medians(J[good], n)).max() for n in SUBSETS)))
    J2 = np.load(IDF / "summary_h0.001_n50_J.npy")
    out["step_max_change"] = float(max(np.abs(medians(J[:50], n) - medians(J2, n)).max() for n in SUBSETS))
    (ROOT / "results" / "checks.json").write_text(json.dumps(out, indent=1))
    print(out)
