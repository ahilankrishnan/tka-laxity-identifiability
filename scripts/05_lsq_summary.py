"""Summary statistics of the least-squares calibration (04_lsq.py) for the manuscript text.

Writes results/lsq/summary.json.  Only converged solves enter any statistic: fits are retained only if their
re-evaluated endpoint converged, and held-out errors are computed on the knees whose held-out predictions
converged for every estimator (the common set is reported as heldout_n_knees).
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.model import Knee                                   # noqa: E402
from knee.sampling import from_vector, PARAM_NAMES            # noqa: E402
from knee.protocols import examine, heldout_kinds             # noqa: E402
from knee.subsets import noise_sd, test_noise, subset_index   # noqa: E402
from knee.analysis import best_fits                           # noqa: E402

L = ROOT / "results" / "lsq"
COST_TOL = 1.0      # fits within this fit error of the best one count as equally good
SEPARATION = 0.2    # normalized parameter distance above which two endpoints are distinct
ESTIMATORS = ("prior_center", "gaps", "nominal", "known", "truth_nominal_geometry")


def max_separation(fits: list[dict]) -> float:
    """Largest coordinate difference between the endpoints of any two fits."""
    return max(np.abs(np.array(a["x"]) - np.array(b["x"])).max() for a in fits for b in fits)


def stage_summary(stage: str, d: dict) -> dict:
    """Fit statistics of one least-squares stage."""
    nmeas = len(subset_index("Gaps 0/90")) if stage == "gaps" else d["y"].shape[1]
    fits = json.loads((L / f"{stage}.json").read_text())["fits"]
    by: dict[int, list[dict]] = {}
    for f in fits:
        by.setdefault(f["i"], []).append(f)
    best = best_fits(fits)
    ids = sorted(best)
    T = d["theta"][ids]
    X = np.array([best[i]["x"] for i in ids])
    amb = sep = 0
    for i in ids:
        conv = [f for f in by[i] if f["ok_end"]]
        low = [f for f in conv if f["cost"] <= best[i]["cost"] + COST_TOL]
        sep += max_separation(conv) > SEPARATION
        amb += len(low) > 1 and max_separation(low) > SEPARATION
    t_fit = np.array([f["time"] for f in fits])
    n_par = len(PARAM_NAMES)
    return dict(
        n_knees=len(ids), limit_all=sum(f["status"] == 0 for f in fits), n_all=len(fits),
        endpoints_not_converged=sum(not f["ok_end"] for f in fits),
        fits_with_failed_evals=sum(f["n_failed_evals"] > 0 for f in fits),
        failed_evals_total=sum(f["n_failed_evals"] for f in fits),
        status_all={s: sum(f["status"] == s for f in fits) for s in range(-1, 5)},
        status_retained={s: sum(best[i]["status"] == s for i in ids) for s in range(-1, 5)},
        cost_median=float(np.median([best[i]["cost"] for i in ids])),
        cost_expected=(nmeas - n_par) / 2 if nmeas > n_par else 0.0,
        rmse_all=float(np.sqrt(((X - T) ** 2).mean())), rmse_param=np.sqrt(((X - T) ** 2).mean(0)).round(3).tolist(),
        knees_sep_gt02=int(sep), knees_ambiguous=int(amb),
        nfev_median=float(np.median([f["nfev"] for f in fits])), nfev_max=int(max(f["nfev"] for f in fits)),
        time_fit_median_s=float(np.median(t_fit)),
        time_knee_median_s=float(np.median([sum(f["time"] for f in by[i]) for i in ids])),
        prior_rmse_param=np.sqrt(((T - 0.5) ** 2).mean(0)).round(3).tolist(),
        prior_rmse_all=float(np.sqrt(((T - 0.5) ** 2).mean())))


def truth_cost(d: dict, Yn: np.ndarray) -> dict:
    """Fit error of the true parameters in the nominal-geometry model, on knees where that solve converged."""
    sd = noise_sd()
    best = best_fits(json.loads((L / "nominal.json").read_text())["fits"])
    ids = sorted(best)
    ct, ok = [], []
    for i in ids:
        y, o = examine(Knee(params=from_vector(d["theta"][i])))
        ct.append(0.5 * (((y - Yn[i]) / sd) ** 2).sum())
        ok.append(bool(o))
    ok = np.array(ok)
    ct = np.array(ct)[ok]
    cl = np.array([best[i]["cost"] for i in ids])[ok]
    return dict(n_knees=len(ids), n_converged=int(ok.sum()), median=float(np.median(ct)),
                ratio_median=float(np.median(ct / cl)), n_lower_ls=int((cl < ct).sum()))


def heldout_summary() -> dict:
    """Pooled held-out RMSE of each estimator on the knees where every estimator's prediction converged."""
    ho = np.load(L / "heldout.npz")
    kinds = heldout_kinds(list(ho["h_names"]))
    common = np.all([ho[f"ok_{e}"] for e in ESTIMATORS], 0) & ho["ok_true"]
    return dict(
        heldout_n_outputs={k: len(v) for k, v in kinds.items()},
        heldout_ok={e: int(ho[f"ok_{e}"].sum()) for e in ESTIMATORS},
        heldout_true_ok=int(ho["ok_true"].sum()),
        heldout_n_knees=int(common.sum()), heldout_knees_excluded=ho["sel"][~common].tolist(),
        heldout_rmse={e: {k: float(np.sqrt(((ho[f"h_{e}"][common][:, idx] - ho["h_true"][common][:, idx]) ** 2).mean()))
                          for k, idx in kinds.items()} for e in ESTIMATORS})


def main() -> None:
    """Write results/lsq/summary.json."""
    z = np.load(ROOT / "results" / "data" / "test_nominal.npz")
    d = {k: z[k] for k in z.files}
    Yn = d["y"] + test_noise(d["y"].shape)
    out = {stage: stage_summary(stage, d) for stage in ("gaps", "nominal", "known")}
    out["truth_nominal_cost"] = truth_cost(d, Yn)
    out.update(heldout_summary())
    (L / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
