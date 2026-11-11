"""Bounded nonlinear least-squares calibration of the test knees and held-out predictions.

Stages (each writes results/lsq/<stage>.json):
  nominal  full examination fitted with the nominal attachment sites (attachment mismatch present);
  known    full examination fitted with each knee's true attachment sites;
  gaps     medial and lateral gaps at 0 and 90 deg only, nominal attachment sites;
  heldout  held-out predictions of the lowest-cost fit per knee (results/lsq/heldout.npz).

All stages use the first N_LS test knees, the same fixed noise realization, the unit-box bounds in
normalized coordinates, and the same three starts per knee (prior center and two random draws).
"""
import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.model import Knee                                   # noqa: E402
from knee.sampling import from_vector                         # noqa: E402
from knee.protocols import examine, heldout, neutral, DISTRACTION, JOINT_LINE_Z  # noqa: E402
from knee.subsets import noise_sd, test_noise, subset_index   # noqa: E402

DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "lsq"
N_LS = 50
MAX_NFEV = 500
GAP_IDX = subset_index("Gaps 0/90")


def gaps_exam(knee: Knee) -> tuple[np.ndarray, bool]:
    """Medial and lateral gaps at 0 and 90 deg (same neutral pose and distraction test as `examine`)."""
    vals, q, allok = [], None, True
    for f in (0.0, 90.0):
        th = np.radians(f)
        q, ok = neutral(knee, th, q)
        dist = [([0, 0, -DISTRACTION], [0, y_, JOINT_LINE_Z], [0, 0, 0]) for y_ in (-22.0, 22.0)]
        qs = q.copy(); qs[2] -= 0.5
        qg, okg = knee.solve_robust(th, dist, q0=qs, fixed=[0, 1, 4])
        allok &= bool(ok and okg)
        vals += list(knee.gaps(qg, th))
    return np.array(vals), allok


def fit_job(args: tuple) -> dict:
    """One bounded least-squares fit from one start."""
    i, y_obs, x0, tag, mode, offs = args
    t0 = time.time()
    if mode == "gaps":
        sd = noise_sd()[GAP_IDX]
        sim = lambda x: gaps_exam(Knee(params=from_vector(x)))
    else:
        sd = noise_sd()
        sim = lambda x: examine(Knee(params=from_vector(x), attach_offsets=offs))
    n_fail = [0]

    def f(x):
        y, ok = sim(x)
        n_fail[0] += not ok          # counted and reported; the endpoint itself must converge (below)
        return y

    r = least_squares(lambda x: (f(x) - y_obs) / sd, x0, bounds=(0, 1), diff_step=1e-2,
                      max_nfev=MAX_NFEV, x_scale=0.2)
    y_end, ok_end = sim(r.x)          # re-evaluate the endpoint: its cost and convergence decide retention
    cost_end = float(0.5 * (((y_end - y_obs) / sd) ** 2).sum())
    return dict(i=i, tag=tag, x=r.x.tolist(), x0=list(map(float, x0)), cost=cost_end, cost_optimizer=float(r.cost),
                ok_end=bool(ok_end), n_failed_evals=int(n_fail[0]),
                status=int(r.status), nfev=int(r.nfev), njev=int(r.njev), time=time.time() - t0)


def starts(sel):
    """Prior center and two random starts in [0.1, 0.9]^8 per knee."""
    rng = np.random.default_rng(7)
    out = []
    for i in sel:
        out.append((int(i), np.full(8, 0.5), "center"))
        for r_ in range(2):
            out.append((int(i), rng.uniform(0.1, 0.9, 8), f"random{r_}"))
    return out


def load():
    z = np.load(DATA / "test_nominal.npz")
    d = {k: z[k] for k in z.files}
    sel = np.where(d["ok"] & d["ok_h"])[0][:N_LS]
    return d, sel


def main(stage: str, nproc: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, sel = load()
    if stage == "heldout":
        return heldout_stage(d, sel, nproc)
    Yn = d["y"] + test_noise(d["y"].shape)
    if stage == "nominal":
        jobs = [(i, Yn[i], x0, tag, "full", None) for i, x0, tag in starts(sel)]
    elif stage == "gaps":
        jobs = [(i, Yn[i][GAP_IDX], x0, tag, "gaps", None) for i, x0, tag in starts(sel)]
    elif stage == "known":
        jobs = [(i, Yn[i], x0, tag, "full", d["offsets"][i]) for i, x0, tag in starts(sel)]
    t0 = time.time()
    with Pool(nproc) as pool:
        res = pool.map(fit_job, jobs, chunksize=1)
    print(stage, f"{len(res)} fits in {time.time() - t0:.0f} s; limit reached:", sum(r["status"] == 0 for r in res),
          "; endpoints not converged:", sum(not r["ok_end"] for r in res),
          "; fits with failed evaluations:", sum(r["n_failed_evals"] > 0 for r in res), flush=True)
    (OUT / f"{stage}.json").write_text(json.dumps(dict(knees=sel.tolist(), max_nfev=MAX_NFEV, fits=res), indent=1))


def best_fits(stage):
    """Lowest re-evaluated fit error per knee among starts whose endpoint converged."""
    best = {}
    for f in json.loads((OUT / f"{stage}.json").read_text())["fits"]:
        if not f["ok_end"]:
            continue
        if f["i"] not in best or f["cost"] < best[f["i"]]["cost"]:
            best[f["i"]] = f
    return best


def hjob(args):
    theta, offs = args
    _, h, ok = heldout(Knee(params=from_vector(np.clip(theta, 0, 1)), attach_offsets=offs))
    return h, ok


def heldout_stage(d, sel, nproc):
    bests = {st: best_fits(st) for st in ("nominal", "gaps", "known")}
    dropped = [int(i) for i in sel if not all(int(i) in b for b in bests.values())]
    sel = np.array([i for i in sel if int(i) not in dropped])
    if dropped:
        print("knees without a converged endpoint in some stage (no held-out predictions):", dropped, flush=True)
    out = dict(sel=sel, h_true=d["h"][sel], ok_true=d["ok_h"][sel], h_names=d["h_names"], theta_true=d["theta"][sel])
    with Pool(nproc) as pool:
        for stage, known in (("nominal", False), ("gaps", False), ("known", True)):
            b = bests[stage]
            th = np.array([b[int(i)]["x"] for i in sel])
            offs = [d["offsets"][i] if known else None for i in sel]
            r = pool.map(hjob, list(zip(th, offs)))
            out[f"theta_{stage}"] = th
            out[f"h_{stage}"] = np.array([a for a, _ in r]); out[f"ok_{stage}"] = np.array([b_ for _, b_ in r])
        for name, th in (("prior_center", np.full((sel.size, 8), 0.5)), ("truth_nominal_geometry", d["theta"][sel])):
            r = pool.map(hjob, [(t, None) for t in th])
            out[f"h_{name}"] = np.array([a for a, _ in r]); out[f"ok_{name}"] = np.array([b_ for _, b_ in r])
    np.savez(OUT / "heldout.npz", **out)
    print("held-out predictions written", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
