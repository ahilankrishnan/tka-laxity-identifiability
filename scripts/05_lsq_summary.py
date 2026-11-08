"""Summary statistics of the least-squares calibration (04_lsq.py) for the manuscript text."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.model import Knee                       # noqa: E402
from knee.sampling import from_vector             # noqa: E402
from knee.protocols import examine                # noqa: E402
from knee.subsets import noise_sd, test_noise     # noqa: E402

L = ROOT / "results" / "lsq"
z = np.load(ROOT / "results" / "data" / "test_nominal.npz")
d = {k: z[k] for k in z.files}
Yn = d["y"] + test_noise(d["y"].shape)
out = {}
for stage, nmeas in (("gaps", 4), ("nominal", 32), ("known", 32)):
    fits = json.loads((L / f"{stage}.json").read_text())["fits"]
    by = {}
    for f in fits:
        by.setdefault(f["i"], []).append(f)
    best = {i: min(v, key=lambda f: f["cost"]) for i, v in by.items()}
    ids = sorted(best)
    T = d["theta"][ids]
    X = np.array([best[i]["x"] for i in ids])
    amb = sep = 0
    for i, v in by.items():
        c0 = best[i]["cost"]
        low = [f for f in v if f["cost"] <= c0 + 1.0]
        far = max(np.abs(np.array(a["x"]) - np.array(b["x"])).max() for a in v for b in v)
        sep += far > 0.2
        if len(low) > 1 and max(np.abs(np.array(a["x"]) - np.array(b["x"])).max() for a in low for b in low) > 0.2:
            amb += 1
    t_fit = np.array([f["time"] for f in fits])
    out[stage] = dict(
        n_knees=len(ids), limit_all=sum(f["status"] == 0 for f in fits), n_all=len(fits),
        status_all={s: sum(f["status"] == s for f in fits) for s in range(-1, 5)},
        status_retained={s: sum(best[i]["status"] == s for i in ids) for s in range(-1, 5)},
        cost_median=float(np.median([best[i]["cost"] for i in ids])), cost_expected=(nmeas - 8) / 2 if nmeas > 8 else 0.0,
        rmse_all=float(np.sqrt(((X - T) ** 2).mean())), rmse_param=np.sqrt(((X - T) ** 2).mean(0)).round(3).tolist(),
        knees_sep_gt02=int(sep), knees_ambiguous=int(amb),
        nfev_median=float(np.median([f["nfev"] for f in fits])), nfev_max=int(max(f["nfev"] for f in fits)),
        time_fit_median_s=float(np.median(t_fit)),
        time_knee_median_s=float(np.median([sum(f["time"] for f in by[i]) for i in ids])))
    out[stage]["prior_rmse_param"] = np.sqrt(((T - 0.5) ** 2).mean(0)).round(3).tolist()
    out[stage]["prior_rmse_all"] = float(np.sqrt(((T - 0.5) ** 2).mean()))
# fit error of the true parameters in the nominal-geometry model (full examination)
sd = noise_sd()
best = {}
for f in json.loads((L / "nominal.json").read_text())["fits"]:
    if f["i"] not in best or f["cost"] < best[f["i"]]["cost"]:
        best[f["i"]] = f
ct = []
for i in sorted(best):
    y, _ = examine(Knee(params=from_vector(d["theta"][i])))
    ct.append(0.5 * (((y - Yn[i]) / sd) ** 2).sum())
ct = np.array(ct); cl = np.array([best[i]["cost"] for i in sorted(best)])
out["truth_nominal_cost"] = dict(median=float(np.median(ct)), ratio_median=float(np.median(ct / cl)), n_lower_ls=int((cl < ct).sum()))
ho = np.load(L / "heldout.npz")
names = list(ho["h_names"])
kinds = {"AP": [i for i, n in enumerate(names) if n.split("@")[0] in ("ant", "post") or n.endswith(":ap@45")],
         "VV": [i for i, n in enumerate(names) if n.split("@")[0] in ("var", "val") or n.endswith(":vv@45")],
         "IE": [i for i, n in enumerate(names) if n.split("@")[0] in ("int", "ext") or n.endswith(":ie@45")]}
E = ("prior_center", "gaps", "nominal", "known", "truth_nominal_geometry")
out["heldout_n_outputs"] = {k: len(v) for k, v in kinds.items()}
out["heldout_ok"] = {e: int(ho[f"ok_{e}"].sum()) for e in E}
out["heldout_true_ok"] = int(ho["ok_true"].sum())
out["heldout_rmse"] = {e: {k: float(np.sqrt(((ho[f"h_{e}"][:, idx] - ho["h_true"][:, idx]) ** 2).mean())) for k, idx in kinds.items()}
                       for e in E}
(L / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
