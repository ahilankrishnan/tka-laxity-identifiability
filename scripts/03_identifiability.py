"""Local (Fisher-information) identifiability of the ligament parameters for each measurement subset.

For a set of test knees, the sensitivity matrix J = dy/dtheta (theta normalized to [0, 1]) is
computed with central differences through the forward simulator.  Combined with the measurement
noise and a Gaussian with the variance of the uniform prior (1/12), the linearized posterior covariance
C = (J^T S^-2 J + 12 I)^-1 gives an identifiability index 1 - sd_post / sd_prior per parameter.
"""
import sys
import json
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.model import Knee                                   # noqa: E402
from knee.sampling import from_vector, PARAM_NAMES            # noqa: E402
from knee.protocols import examine                            # noqa: E402
from knee.subsets import SUBSETS, subset_index, noise_sd      # noqa: E402

DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "identifiability"
N_KNEES = 500
H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02     # central-difference step (normalized)
N_USE = int(sys.argv[2]) if len(sys.argv) > 2 else N_KNEES


def jac_job(args: tuple) -> tuple[np.ndarray, bool]:
    """Central-difference sensitivity matrix of the 32 outputs to the 8 normalized parameters."""
    theta, offsets = args
    J = np.zeros((32, 8))
    ok = True
    for j in range(8):
        cols = []
        for s in (+1, -1):
            t = theta.copy()
            t[j] = t[j] + s * H
            y, o = examine(Knee(params=from_vector(t), attach_offsets=offsets))
            ok &= o
            cols.append(y)
        J[:, j] = (cols[0] - cols[1]) / (2 * H)
    return J, ok


def posterior(J: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fisher information and linearized posterior covariance for the outputs `idx`."""
    S = noise_sd()[idx]
    A = J[idx] / S[:, None]
    F = A.T @ A
    C = np.linalg.inv(F + 12.0 * np.eye(8))
    return F, C


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA / "test_nominal.npz")
    sel = np.where(d["ok"])[0][:N_USE]
    # keep the interior of the prior so the central differences stay inside it
    theta = np.clip(d["theta"][sel], H, 1 - H)
    with Pool(4) as pool:
        r = pool.map(jac_job, [(theta[i], d["offsets"][k]) for i, k in enumerate(sel)])
    Js = np.array([a for a, _ in r])
    summary = {}
    for name in SUBSETS:
        idx = subset_index(name)
        ident, corr, eig = [], [], []
        for J in Js:
            F, C = posterior(J, idx)
            sd = np.sqrt(np.diag(C))
            ident.append(1 - sd / np.sqrt(1 / 12))
            corr.append(C / np.outer(sd, sd))
            eig.append(np.sort(np.linalg.eigvalsh(F))[::-1])
        summary[name] = dict(ident_median=np.median(ident, 0).tolist(), corr_median=np.median(corr, 0).tolist(),
                             eig_median=np.median(eig, 0).tolist())
        print(f"{name:12s}", np.round(summary[name]["ident_median"], 2), flush=True)
    name = "summary.json" if (H == 0.02 and N_USE == N_KNEES) else f"summary_h{H:g}_n{N_USE}.json"
    (OUT / name).write_text(json.dumps(dict(params=PARAM_NAMES, n_knees=int(sel.size), step=H, subsets=summary), indent=1))
    np.save(OUT / name.replace(".json", "_J.npy"), Js)


if __name__ == "__main__":
    main()
