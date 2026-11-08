"""Generate the synthetic virtual-knee populations.

Characterization knees (train.npz, 10000) and test knees (test_nominal.npz, 500) vary the eight ligament
parameters and the ligament attachment sites (1.5 mm SD per coordinate), each knee from its own seed.
Test knees also carry the held-out outputs.  Measurement noise is added at inference time.
"""
import sys
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.model import Knee, INSERT                       # noqa: E402
from knee.sampling import sample_params, sample_offsets, to_vector, PARAM_NAMES  # noqa: E402
from knee.protocols import examine, heldout, feature_names  # noqa: E402

OUT = ROOT / "results" / "data"
N_TRAIN, N_TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 10000, int(sys.argv[2]) if len(sys.argv) > 2 else 500
TEST_SEED0 = 100_000
NPROC = 4

MISMATCH = {
    "nominal": {},
}


def _knee(seed: int, attach_sd: float = 1.5, **kw) -> tuple[Knee, dict]:
    """Virtual knee with parameters and attachment offsets drawn from independent seeds."""
    rng = np.random.default_rng(seed)
    p = sample_params(rng, 1)[0]
    off_rng = np.random.default_rng(seed + 7_000_000)      # offsets drawn independently of params
    return Knee(params=p, attach_offsets=sample_offsets(off_rng, attach_sd), **kw), p


def train_job(seed: int):
    """Full examination of one training knee."""
    k, p = _knee(seed)
    y, ok = examine(k)
    return to_vector(p), y, ok


def test_job(args: tuple) -> dict:
    """Full examination of one test knee under a mismatch case (plus held-out loads for nominal)."""
    seed, case = args
    kw = dict(MISMATCH[case])
    k, p = _knee(seed, **kw)
    y, ok = examine(k)
    res = dict(theta=to_vector(p), y=y, ok=ok)
    if case == "nominal":
        names, h, okh = heldout(k)
        res.update(h=h, ok_h=okh, offsets=k.attach_offsets)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(NPROC) as pool:
        for case in MISMATCH:
            r = pool.map(test_job, [(TEST_SEED0 + i, case) for i in range(N_TEST)], chunksize=2)
            d = dict(theta=np.array([x["theta"] for x in r]), y=np.array([x["y"] for x in r]),
                     ok=np.array([x["ok"] for x in r]), features=feature_names(), params=PARAM_NAMES)
            if case == "nominal":
                d.update(h=np.array([x["h"] for x in r]), ok_h=np.array([x["ok_h"] for x in r]),
                         offsets=np.array([x["offsets"] for x in r]))
                d["h_names"] = heldout(Knee())[0]
            np.savez(OUT / f"test_{case}.npz", **d)
            print(f"test {case}: {d['ok'].sum()}/{N_TEST} converged, {time.time() - t0:.0f} s", flush=True)

        r = pool.map(train_job, range(N_TRAIN), chunksize=8)
        th = np.array([a for a, _, _ in r]); Y = np.array([b for _, b, _ in r]); ok = np.array([c for *_, c in r])
        np.savez(OUT / "train.npz", theta=th, y=Y, ok=ok, features=feature_names(), params=PARAM_NAMES)
        print(f"train: {ok.sum()}/{ok.size} converged, {time.time() - t0:.0f} s", flush=True)

if __name__ == "__main__":
    main()
