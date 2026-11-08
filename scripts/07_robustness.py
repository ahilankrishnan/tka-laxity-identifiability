"""Supplementary robustness summaries from saved results (no new simulations).

1. Fisher identifiability with doubled measurement-noise standard deviations, from the saved sensitivity
   matrices of the main analysis, and whether the ranking of the measurement sets persists.
2. Across-knee median and interquartile range of the held-out prediction errors of each estimator.
Writes results/robustness.json.
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.subsets import SUBSETS, subset_index, noise_sd      # noqa: E402

RES = ROOT / "results"


def indices(Js: np.ndarray, idx: np.ndarray, scale: float) -> np.ndarray:
    """Median identifiability index of each parameter with noise sd multiplied by `scale`."""
    S = noise_sd()[idx] * scale
    out = []
    for J in Js:
        A = J[idx] / S[:, None]
        C = np.linalg.inv(A.T @ A + 12.0 * np.eye(8))
        out.append(1 - np.sqrt(np.diag(C)) / np.sqrt(1 / 12))
    return np.median(out, 0)


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-x))


def main():
    Js = np.load(RES / "identifiability" / "summary_J.npy")
    names = list(SUBSETS)
    base = np.array([indices(Js, subset_index(n), 1.0) for n in names])     # subsets x parameters
    dbl = np.array([indices(Js, subset_index(n), 2.0) for n in names])
    mean_b, mean_d = base.mean(1), dbl.mean(1)
    rho = [float(np.corrcoef(ranks(base[:, j]), ranks(dbl[:, j]))[0, 1]) for j in range(8)]
    ident = dict(subsets=names, mean_index_base=mean_b.round(3).tolist(), mean_index_2x=mean_d.round(3).tolist(),
                 rank_base=ranks(mean_b).tolist(), rank_2x=ranks(mean_d).tolist(),
                 spearman_per_parameter=np.round(rho, 3).tolist(),
                 index_2x=dbl.round(3).tolist(), index_base=base.round(3).tolist())
    ho = np.load(RES / "lsq" / "heldout.npz")
    hn = list(ho["h_names"])
    kinds = {"AP": [i for i, n in enumerate(hn) if n.split("@")[0] in ("ant", "post") or n.endswith(":ap@45")],
             "VV": [i for i, n in enumerate(hn) if n.split("@")[0] in ("var", "val") or n.endswith(":vv@45")],
             "IE": [i for i, n in enumerate(hn) if n.split("@")[0] in ("int", "ext") or n.endswith(":ie@45")]}
    held = {}
    for e in ("prior_center", "gaps", "nominal", "known", "truth_nominal_geometry"):
        held[e] = {}
        for k, idx in kinds.items():
            per_knee = np.sqrt(((ho[f"h_{e}"][:, idx] - ho["h_true"][:, idx]) ** 2).mean(1))
            q1, med, q3 = np.percentile(per_knee, [25, 50, 75])
            held[e][k] = dict(median=round(float(med), 2), q1=round(float(q1), 2), q3=round(float(q3), 2),
                              pooled=round(float(np.sqrt(((ho[f"h_{e}"][:, idx] - ho["h_true"][:, idx]) ** 2).mean())), 2))
    out = dict(identifiability_2x_noise=ident, heldout_per_knee=held)
    (RES / "robustness.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(ident["mean_index_base"]), json.dumps(ident["mean_index_2x"]))
    print("ranks", ident["rank_base"], ident["rank_2x"], "spearman", ident["spearman_per_parameter"])
    for e, v in held.items():
        print(e, v)


if __name__ == "__main__":
    main()
