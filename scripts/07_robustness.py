"""Supplementary robustness summaries from saved results (no new simulations).

1. Fisher identifiability with doubled measurement-noise standard deviations, from the saved sensitivity
   matrices of the main analysis, and whether the ranking of the measurement sets persists.
2. Across-knee median and interquartile range of the held-out prediction errors of each estimator, on the
   knees where every estimator's held-out prediction converged (as in 05_lsq_summary.py).
Writes results/robustness.json.
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.subsets import SUBSETS, subset_index                # noqa: E402
from knee.protocols import heldout_kinds                      # noqa: E402
from knee.analysis import median_index                        # noqa: E402

RES = ROOT / "results"
NOISE_FACTOR = 2.0    # multiplier of the measurement-noise standard deviations in the robustness check
ESTIMATORS = ("prior_center", "gaps", "nominal", "known", "truth_nominal_geometry")


def ranks(x: np.ndarray) -> np.ndarray:
    """Rank of each entry (0 = largest)."""
    return np.argsort(np.argsort(-x))


def main() -> None:
    """Write results/robustness.json."""
    Js = np.load(RES / "identifiability" / "summary_J.npy")
    names = list(SUBSETS)
    base = np.array([median_index(Js, subset_index(n), 1.0) for n in names])     # subsets x parameters
    dbl = np.array([median_index(Js, subset_index(n), NOISE_FACTOR) for n in names])
    mean_b, mean_d = base.mean(1), dbl.mean(1)
    rho = [float(np.corrcoef(ranks(base[:, j]), ranks(dbl[:, j]))[0, 1]) for j in range(8)]
    ident = dict(subsets=names, mean_index_base=mean_b.round(3).tolist(), mean_index_2x=mean_d.round(3).tolist(),
                 rank_base=ranks(mean_b).tolist(), rank_2x=ranks(mean_d).tolist(),
                 spearman_per_parameter=np.round(rho, 3).tolist(),
                 index_2x=dbl.round(3).tolist(), index_base=base.round(3).tolist())
    ho = np.load(RES / "lsq" / "heldout.npz")
    kinds = heldout_kinds(list(ho["h_names"]))
    common = np.all([ho[f"ok_{e}"] for e in ESTIMATORS], 0) & ho["ok_true"]
    true = ho["h_true"][common]
    held = {}
    for e in ESTIMATORS:
        h = ho[f"h_{e}"][common]
        held[e] = {}
        for k, idx in kinds.items():
            per_knee = np.sqrt(((h[:, idx] - true[:, idx]) ** 2).mean(1))
            q1, med, q3 = np.percentile(per_knee, [25, 50, 75])
            held[e][k] = dict(median=round(float(med), 2), q1=round(float(q1), 2), q3=round(float(q3), 2),
                              pooled=round(float(np.sqrt(((h[:, idx] - true[:, idx]) ** 2).mean())), 2))
    out = dict(identifiability_2x_noise=ident, heldout_n_knees=int(common.sum()), heldout_per_knee=held)
    (RES / "robustness.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(ident["mean_index_base"]), json.dumps(ident["mean_index_2x"]))
    print("ranks", ident["rank_base"], ident["rank_2x"], "spearman", ident["spearman_per_parameter"])
    for e, v in held.items():
        print(e, v)


if __name__ == "__main__":
    main()
