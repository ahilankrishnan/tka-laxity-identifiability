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
from knee.protocols import examine, heldout, neutral, gap_test, FLEX_TRAIN  # noqa: E402
from knee.subsets import noise_sd, test_noise, subset_index   # noqa: E402
from knee.analysis import best_fits                           # noqa: E402

DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "lsq"
N_LS = 50            # test knees fitted
MAX_NFEV = 500       # cap on SciPy's nfev counter per start; it excludes the model calls of the finite-difference
                     # Jacobians and of the endpoint re-evaluation, which are counted in `model_evaluations`
DIFF_STEP = 1e-2     # relative finite-difference step of the least-squares Jacobian
X_SCALE = 0.2        # characteristic scale of the normalized parameters for the trust region
START_SEED = 7       # random starts
START_RANGE = (0.1, 0.9)   # random starts are drawn inside the prior box, away from its bounds
N_RANDOM_STARTS = 2
GAP_IDX = subset_index("Gaps 0/90")


class EquilibriumFailure(RuntimeError):
    """Abort an optimization start rather than fit outputs from a failed equilibrium solve."""


def gaps_exam(knee: Knee) -> tuple[np.ndarray, bool]:
    """Gaps at 0 and 90 deg, following the full examination's neutral checkpoints (FLEX_TRAIN)."""
    vals, q, allok, thp = [], None, True, None
    for f in FLEX_TRAIN:
        th = np.radians(f)
        q, ok = neutral(knee, th, q, thp)
        thp = th
        allok &= bool(ok)
        if f in (0.0, 90.0):
            g, okg = gap_test(knee, th, q)
            allok &= bool(okg)
            vals += list(g)
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
    n_eval = 0
    last_x = np.array(x0, float)

    def f(x: np.ndarray) -> np.ndarray:
        """Use only converged finite simulations in the optimizer, including its finite differences."""
        nonlocal n_eval, last_x
        n_eval += 1
        last_x = x.copy()
        y, ok = sim(x)
        if not ok or not np.all(np.isfinite(y)):
            raise EquilibriumFailure("A model evaluation did not converge; this start was rejected.")
        return y

    try:
        r = least_squares(lambda x: (f(x) - y_obs) / sd, x0, bounds=(0, 1), diff_step=DIFF_STEP,
                          max_nfev=MAX_NFEV, x_scale=X_SCALE)
        y_end = f(r.x)              # re-evaluate by exactly the same acceptance rule
    except EquilibriumFailure as exc:
        return dict(i=i, tag=tag, x=last_x.tolist(), x0=list(map(float, x0)), cost=None,
                    cost_optimizer=None, ok_end=False, n_failed_evals=1, status=-1,
                    nfev=None, njev=None, model_evaluations=n_eval,
                    message=str(exc), time=time.time() - t0)
    cost_end = float(0.5 * (((y_end - y_obs) / sd) ** 2).sum())
    return dict(i=i, tag=tag, x=r.x.tolist(), x0=list(map(float, x0)), cost=cost_end, cost_optimizer=float(r.cost),
                ok_end=True, n_failed_evals=0, model_evaluations=n_eval, message=str(r.message),
                status=int(r.status), nfev=int(r.nfev), njev=int(r.njev), time=time.time() - t0)


def starts(sel: np.ndarray) -> list[tuple[int, np.ndarray, str]]:
    """Prior center and N_RANDOM_STARTS random starts in START_RANGE^8 per knee."""
    rng = np.random.default_rng(START_SEED)
    out = []
    for i in sel:
        out.append((int(i), np.full(8, 0.5), "center"))
        for r_ in range(N_RANDOM_STARTS):
            out.append((int(i), rng.uniform(*START_RANGE, 8), f"random{r_}"))
    return out


def load() -> tuple[dict, np.ndarray]:
    """Test-knee data and the first N_LS knees whose examination and held-out outputs converged."""
    z = np.load(DATA / "test_nominal.npz")
    d = {k: z[k] for k in z.files}
    sel = np.where(d["ok"] & d["ok_h"])[0][:N_LS]
    if sel.size < N_LS:
        raise RuntimeError(f"Only {sel.size} accepted knees are available; {N_LS} are required for calibration.")
    return d, sel


def main(stage: str, nproc: int) -> None:
    """Run the fits of one stage (or the held-out predictions) on nproc processes."""
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
          "; rejected starts:", sum(not r["ok_end"] for r in res),
          "; fits with failed evaluations:", sum(r["n_failed_evals"] > 0 for r in res), flush=True)
    (OUT / f"{stage}.json").write_text(json.dumps(dict(knees=sel.tolist(), max_nfev=MAX_NFEV, fits=res), indent=1))


def hjob(args: tuple) -> tuple[np.ndarray, bool]:
    """Held-out outputs and convergence flag for one parameter vector and attachment offsets."""
    theta, offs = args
    _, h, ok = heldout(Knee(params=from_vector(np.clip(theta, 0, 1)), attach_offsets=offs))
    return h, ok


def heldout_stage(d: dict, sel: np.ndarray, nproc: int) -> None:
    """Held-out predictions of the best fit of each stage, the prior center, and the true parameters."""
    bests = {st: best_fits(json.loads((OUT / f"{st}.json").read_text())["fits"]) for st in ("nominal", "gaps", "known")}
    # every comparison uses the same fixed knee cohort, so a knee without an accepted fit in any stage stops the
    # pipeline instead of silently shrinking the cohort of some statistics but not of others
    dropped = {st: [int(i) for i in sel if int(i) not in b] for st, b in bests.items()}
    if any(dropped.values()):
        raise RuntimeError("Knees without an accepted fit (inspect results/lsq/<stage>.json): "
                           + "; ".join(f"{st}: {v}" for st, v in dropped.items() if v))
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
