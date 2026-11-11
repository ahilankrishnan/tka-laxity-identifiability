"""Numerical checks reported in the Supplementary Material.

1. Path dependence: outputs with the default loading path (INCREMENTS[0] increments) versus every load
   applied in 20 increments, (a) for the examinations of test knees 40-119 with their generating parameters
   and (b) for the held-out predictions of all retained fits; counts outputs that differ by more than
   0.05 mm or deg.
2. Sensitivity outliers: valid sensitivity matrices with an absolute noise-normalized sensitivity above 300,
   and the largest change of the median identifiability indices without them.
3. Step check: largest change of the median indices between steps 0.02 and 0.001 on the same knees and
   the same (clipped) evaluation points.
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import knee.model as M                                        # noqa: E402
from knee.sampling import from_vector                         # noqa: E402
from knee.protocols import examine, heldout                   # noqa: E402
from knee.subsets import SUBSETS, subset_index, noise_sd      # noqa: E402

z = np.load(ROOT / "results" / "data" / "test_nominal.npz")
D = {k: z[k] for k in ("theta", "offsets")}
IDF = ROOT / "results" / "identifiability"
LSQ = ROOT / "results" / "lsq"
FINE = 20


def fine(self, theta, load=None, q0=None, fixed=None):
    for steps in (FINE, 2 * FINE):
        q, ok = self.solve(theta, load, q0, fixed, steps)
        if ok:
            break
    return q, ok


def both(fn, knee):
    """Outputs and flags of fn(knee) with the default and with the fine loading path."""
    default = M.Knee.solve_robust
    a = fn(knee)
    M.Knee.solve_robust = fine
    try:
        b = fn(knee)
    finally:
        M.Knee.solve_robust = default
    return a, b


def job_exam(i):
    (y1, ok1), (y2, ok2) = both(examine, M.Knee(params=from_vector(D["theta"][i]), attach_offsets=D["offsets"][i]))
    return np.abs(y1 - y2), bool(ok1 and ok2)


def job_held(args):
    theta, offs = args
    (_, h1, ok1), (_, h2, ok2) = both(heldout, M.Knee(params=from_vector(np.clip(theta, 0, 1)), attach_offsets=offs))
    return np.abs(h1 - h2), bool(ok1 and ok2)


def medians(Js, name):
    idx = subset_index(name); s = noise_sd()[idx]; out = []
    for J in Js:
        a = J[idx] / s[:, None]
        C = np.linalg.inv(a.T @ a + 12 * np.eye(8))
        out.append(1 - np.sqrt(np.diag(C)) / np.sqrt(1 / 12))
    return np.median(out, 0)


if __name__ == "__main__":
    out = {}
    ho = np.load(LSQ / "heldout.npz")
    held_jobs = []
    for stage, known in (("nominal", False), ("gaps", False), ("known", True)):
        for th, i in zip(ho[f"theta_{stage}"], ho["sel"]):
            held_jobs.append((th, D["offsets"][i] if known else None))
    with Pool(4) as p:
        r = p.map(job_exam, range(40, 120))
        rh = p.map(job_held, held_jobs)
    d = np.array([a for a, _ in r]); dh = np.array([a for a, _ in rh])
    out.update(path_knees=80, path_outputs=int(d.size), path_changed=int((d > 0.05).sum()),
               path_max_change=float(d.max()), path_all_converged=all(o for _, o in r),
               path_fitted_predictions=len(rh), path_fitted_outputs=int(dh.size),
               path_fitted_changed=int((dh > 0.05).sum()), path_fitted_max_change=float(dh.max()),
               path_fitted_all_converged=all(o for _, o in rh))
    S = json.loads((IDF / "summary.json").read_text())
    J = np.load(IDF / "summary_J.npy")
    m = (np.abs(J) / noise_sd()[None, :, None]).max(axis=(1, 2))
    bad = np.where(m > 300)[0]; good = np.setdiff1d(np.arange(len(J)), bad)
    out.update(fisher_valid=len(J), fisher_rejected=S["rejected_knees"], fisher_clipped=S["n_clipped"],
               fd_outlier_knees=int(bad.size),
               fd_outlier_max_change=float(max(np.abs(medians(J, n) - medians(J[good], n)).max() for n in SUBSETS)))
    S2 = json.loads((IDF / "summary_h0.001_n50.json").read_text())
    J2 = np.load(IDF / "summary_h0.001_n50_J.npy")
    common = sorted(set(S["knees"]) & set(S2["knees"]))
    i1 = [S["knees"].index(k) for k in common]; i2 = [S2["knees"].index(k) for k in common]
    out["step_knees"] = len(common)
    out["step_max_change"] = float(max(np.abs(medians(J[i1], n) - medians(J2[i2], n)).max() for n in SUBSETS))
    (ROOT / "results" / "checks.json").write_text(json.dumps(out, indent=1))
    print(out)
