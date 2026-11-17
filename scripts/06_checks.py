"""Numerical checks reported in the Supplementary Material.

1. Loading-path refinement: outputs with the default loading procedure versus one twice as fine (half
   the coordinate limits and initial, maximum, and minimum loading increments for every solve),
   (a) for the examinations of test knees 40-119 with their generating parameters and (b) for the held-out
   predictions of all retained fits; counts outputs that differ by more than BRANCH_TOL mm or deg.
2. Sensitivity outliers: valid sensitivity matrices with an absolute noise-normalized sensitivity above
   FD_OUTLIER, and the largest change of the median identifiability indices without them.
3. Step check: largest change of the median indices between steps 0.02 and 0.001 on the same knees and
   the same evaluation points (each knee's own parameters).
"""
import sys
import json
from pathlib import Path
from typing import Callable
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import knee.model as M                                        # noqa: E402
from knee.sampling import from_vector                         # noqa: E402
from knee.protocols import examine, heldout                   # noqa: E402
from knee.subsets import SUBSETS, subset_index, noise_sd      # noqa: E402
from knee.analysis import median_index                        # noqa: E402

IDF = ROOT / "results" / "identifiability"
LSQ = ROOT / "results" / "lsq"
PATH_KNEES = range(40, 120)   # test knees of the path-dependence check (examinations)
BRANCH_TOL = 0.05     # mm or deg; a larger change between paths counts as a changed output
FD_OUTLIER = 300.0    # noise-normalized sensitivity above which a matrix counts as an outlier


def both(fn: Callable[[M.Knee], tuple], knee: M.Knee) -> tuple:
    """Outputs and flags of fn(knee) with the default and with a twice finer loading path."""
    dq, s0, smin = M.PATH_DQ.copy(), M.PATH_STEP0, M.PATH_STEP_MIN
    a = fn(knee)
    M.PATH_DQ, M.PATH_STEP0, M.PATH_STEP_MIN = dq / 2, s0 / 2, smin / 2
    try:
        b = fn(knee)
    finally:
        M.PATH_DQ, M.PATH_STEP0, M.PATH_STEP_MIN = dq, s0, smin
    return a, b


def job_exam(args: tuple) -> tuple[np.ndarray, tuple[bool, bool]]:
    """Output changes between the two paths for one test knee's examination (zeros unless both converge)."""
    theta, offs = args
    (y1, ok1), (y2, ok2) = both(examine, M.Knee(params=from_vector(theta), attach_offsets=offs))
    return np.abs(y1 - y2) if (ok1 and ok2) else np.zeros_like(y1), (bool(ok1), bool(ok2))


def job_held(args: tuple) -> tuple[np.ndarray, tuple[bool, bool]]:
    """Output changes between the two paths for one fitted parameter set's held-out predictions."""
    theta, offs = args
    (_, h1, ok1), (_, h2, ok2) = both(heldout, M.Knee(params=from_vector(np.clip(theta, 0, 1)), attach_offsets=offs))
    return np.abs(h1 - h2) if (ok1 and ok2) else np.zeros_like(h1), (bool(ok1), bool(ok2))


def medians(Js: np.ndarray, name: str) -> np.ndarray:
    """Median identifiability indices of measurement subset `name`."""
    return median_index(Js, subset_index(name))


def main() -> None:
    """Write results/checks.json."""
    D = np.load(ROOT / "results" / "data" / "test_nominal.npz")
    out = {}
    ho = np.load(LSQ / "heldout.npz")
    held_jobs = []
    for stage, known in (("nominal", False), ("gaps", False), ("known", True)):
        for th, i in zip(ho[f"theta_{stage}"], ho["sel"]):
            held_jobs.append((th, D["offsets"][i] if known else None))
    with Pool(4) as p:
        r = p.map(job_exam, [(D["theta"][i], D["offsets"][i]) for i in PATH_KNEES])
        rh = p.map(job_held, held_jobs)
    # only pairs in which both paths converged were compared; the zero placeholders of the others are dropped
    d = np.array([a for a, o in r if all(o)]); dh = np.array([a for a, o in rh if all(o)])
    if d.size == 0 or dh.size == 0:
        raise RuntimeError("Loading-path check: no examination or no held-out prediction converged on both paths.")
    out.update(path_knees=len(PATH_KNEES), path_outputs=int(d.size), path_changed=int((d > BRANCH_TOL).sum()),
               path_max_change=float(d.max()), path_converged_both=len(d),
               path_skipped=len(r) - len(d), path_flag_disagree=sum(o[0] != o[1] for _, o in r),
               path_fitted_predictions=len(rh), path_fitted_outputs=int(dh.size),
               path_fitted_changed=int((dh > BRANCH_TOL).sum()), path_fitted_max_change=float(dh.max()),
               path_fitted_converged_both=len(dh), path_fitted_skipped=len(rh) - len(dh),
               path_fitted_flag_disagree=sum(o[0] != o[1] for _, o in rh))
    S = json.loads((IDF / "summary.json").read_text())
    J = np.load(IDF / "summary_J.npy")
    m = (np.abs(J) / noise_sd()[None, :, None]).max(axis=(1, 2))
    bad = np.where(m > FD_OUTLIER)[0]; good = np.setdiff1d(np.arange(len(J)), bad)
    if good.size == 0:
        raise RuntimeError("Every sensitivity matrix is an outlier; the outlier check has nothing to compare with.")
    out.update(fisher_valid=len(J), fisher_rejected=S["rejected_knees"], fisher_outside=S["n_outside"],
               fd_outlier_knees=int(bad.size),
               fd_outlier_max_change=float(max(np.abs(medians(J, n) - medians(J[good], n)).max() for n in SUBSETS)))
    S2 = json.loads((IDF / "summary_h0.001_n50.json").read_text())
    J2 = np.load(IDF / "summary_h0.001_n50_J.npy")
    common = sorted(set(S["knees"]) & set(S2["knees"]))
    i1 = [S["knees"].index(k) for k in common]; i2 = [S2["knees"].index(k) for k in common]
    if not common:
        raise RuntimeError("The two finite-difference steps share no valid knee; the step check cannot be computed.")
    out["step_knees"] = len(common)
    out["step_max_change"] = float(max(np.abs(medians(J[i1], n) - medians(J2[i2], n)).max() for n in SUBSETS))
    (ROOT / "results" / "checks.json").write_text(json.dumps(out, indent=1, allow_nan=False))
    print(out)


if __name__ == "__main__":
    main()
