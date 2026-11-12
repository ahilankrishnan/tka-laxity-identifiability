"""Quick cross-platform check of the forward model (minutes, not hours).

Rebuilds test knees from their seeds (same draws as 01_generate_data.py), runs the full examination and
the held-out load cases, and saves results/platform_check.npz.  Run it on two computers and compare
the files with scripts/10_compare_runs.py.

Usage: python scripts/11_platform_check.py [first last [extra ids ...]]   (default: knees 0-39 plus the
knees whose outputs differed between computers with earlier solver versions)
"""
import sys
import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.sampling import knee_from_seed, to_vector, TEST_SEED0  # noqa: E402
from knee.protocols import examine, heldout                    # noqa: E402

FLIPPED = [8, 24, 27, 37, 40, 66, 132, 160, 214, 235, 262, 293, 354, 359, 366, 376, 397, 445, 453, 483]


def job(i: int) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray, bool]:
    """Parameters, examination outputs, and held-out outputs (with flags) of test knee i."""
    k, p = knee_from_seed(TEST_SEED0 + i)
    y, ok = examine(k)
    _, h, okh = heldout(k)
    return to_vector(p), y, ok, h, okh


def main() -> None:
    """Simulate the selected test knees and write results/platform_check.npz."""
    a = sys.argv[1:]
    ids = list(range(int(a[0]), int(a[1]))) + [int(x) for x in a[2:]] if a else list(range(40)) + FLIPPED
    t0 = time.time()
    with Pool(4) as pool:
        r = pool.map(job, ids, chunksize=1)
    out = dict(ids=np.array(ids), theta=np.array([x[0] for x in r]), y=np.array([x[1] for x in r]),
               ok=np.array([x[2] for x in r]), h=np.array([x[3] for x in r]), ok_h=np.array([x[4] for x in r]))
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / "platform_check.npz", **out)
    print(f"{len(ids)} knees in {time.time() - t0:.0f} s; converged {out['ok'].sum()} exam, {out['ok_h'].sum()} held-out")


if __name__ == "__main__":
    main()
