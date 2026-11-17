"""Compare two complete result directories (e.g. runs on two computers), or two result files of one type.

Usage: python scripts/10_compare_runs.py RESULTS_A RESULTS_B [--tol 1e-6]
       python scripts/10_compare_runs.py A/platform_check.npz B/platform_check.npz

Only .npz, .npy, and .json files are compared (environment.txt and other text files are not; check the
source digests in environment.txt separately, see README).  Every such file found in EITHER directory is compared; a file present on one side only is a
difference.  In .npz archives the key sets must match; arrays must match in shape and dtype kind before their
values are compared.  In .json files key presence, container types, and every leaf are compared.  NaN and
infinity are compared by position (a NaN or an infinity on one side only, or infinities of different sign,
is a difference); paired NaNs at the same position count as equal.  Timings (SKIP_JSON) are not compared.
Differences above --tol are listed; differences above BRANCH_TOL (mm, deg, or normalized units) are flagged
as possible equilibrium-branch changes.  Exit status 1 if anything differs, and 2 if the inputs are invalid
(missing paths, mixed or unsupported file types, or nothing to compare).
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Iterator, NoReturn
import numpy as np

BRANCH_TOL = 0.05   # mm, deg, or normalized units: a larger difference suggests a different equilibrium branch
SKIP_JSON = ("time", "time_s", "time_fit_median_s", "time_knee_median_s")   # timings differ between computers
SUFFIXES = (".npz", ".npy", ".json")
MISSING = object()   # sentinel for a JSON key that is absent (distinct from an explicit null)


def flat_json(x: Any, key: str = "") -> Iterator[tuple[str, Any]]:
    """(dotted key, leaf value) pairs of a nested JSON object; containers also yield their type and size."""
    if isinstance(x, dict):
        yield key, ("<dict>", len(x))
        for k, v in x.items():
            yield from flat_json(v, f"{key}.{k}" if key else str(k))
    elif isinstance(x, list):
        yield key, ("<list>", len(x))
        for i, v in enumerate(x):
            yield from flat_json(v, f"{key}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield key, float(x)
    else:
        yield key, x


def skipped(key: str) -> bool:
    """True for timing keys, which differ between computers by design."""
    return key.split(".")[-1].split("[")[0] in SKIP_JSON


def array_diff(xa: np.ndarray, xb: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Largest finite difference, the elementwise difference (inf where non-finite values disagree), and the
    number of non-finite mismatches between two float arrays of equal shape."""
    xa, xb = xa.astype(float), xb.astype(float)
    na, nb = np.isnan(xa), np.isnan(xb)
    ia, ib = np.isinf(xa), np.isinf(xb)
    mismatch = (na != nb) | (ia != ib) | (ia & ib & (np.sign(xa) != np.sign(xb)))
    fin = np.isfinite(xa) & np.isfinite(xb)
    d = np.zeros(xa.shape)
    d[fin] = np.abs(xa[fin] - xb[fin])
    d[mismatch] = np.inf
    worst_finite = float(d[fin].max()) if fin.any() else 0.0
    return worst_finite, d, int(mismatch.sum())


def load_arrays(p: Path) -> dict[str, np.ndarray]:
    """Arrays of an .npz archive (by key) or of an .npy file (key '')."""
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    return {"": np.asarray(np.load(p, allow_pickle=False))}


def compare_file(pa: Path, pb: Path, label: str, tol: float) -> tuple[float, int, int]:
    """Compare one pair of files; returns (largest finite difference, differing items, branch entries)."""
    worst, bad, branch = 0.0, 0, 0
    pairs = []
    if pa.suffix == ".json":
        ja, jb = dict(flat_json(json.loads(pa.read_text()))), dict(flat_json(json.loads(pb.read_text())))
        for k in sorted(set(ja) | set(jb)):
            if skipped(k):
                continue
            va, vb = ja.get(k, MISSING), jb.get(k, MISSING)
            if va is MISSING or vb is MISSING:
                side = "A" if va is MISSING else "B"
                print(f"{label}:{k}: key missing in {side}"); bad += 1
            elif isinstance(va, float) and isinstance(vb, float):
                pairs.append((k, np.array([va]), np.array([vb])))
            elif type(va) is not type(vb) or va != vb:
                print(f"{label}:{k}: {va!r} != {vb!r}"); bad += 1
    else:
        za, zb = load_arrays(pa), load_arrays(pb)
        for k in sorted(set(za) ^ set(zb)):
            print(f"{label}:{k}: array missing in {'B' if k in za else 'A'}"); bad += 1
        for k in sorted(set(za) & set(zb)):
            xa, xb = za[k], zb[k]
            if xa.shape != xb.shape:
                print(f"{label}:{k}: shape {xa.shape} != {xb.shape}"); bad += 1; continue
            fa, fb = xa.dtype.kind in "fc", xb.dtype.kind in "fc"
            if fa != fb or (not fa and xa.dtype.kind != xb.dtype.kind):
                print(f"{label}:{k}: dtype {xa.dtype} != {xb.dtype}"); bad += 1; continue
            if fa:
                pairs.append((k, np.real(xa) if xa.dtype.kind == "c" else xa,
                              np.real(xb) if xb.dtype.kind == "c" else xb))
                if xa.dtype.kind == "c" or xb.dtype.kind == "c":
                    pairs.append((k + ".imag", np.imag(xa), np.imag(xb)))
            elif not np.array_equal(xa, xb):
                print(f"{label}:{k}: values differ"); bad += 1
    for k, xa, xb in pairs:
        m, d, nonfinite = array_diff(xa, xb)
        worst = max(worst, m)
        if nonfinite:
            print(f"{label}:{k}: {nonfinite} NaN/inf entries on one side only or of different sign"); bad += 1
        if m > tol:
            bad += 1
            nb = int(((d > BRANCH_TOL) & np.isfinite(d)).sum()); branch += nb
            dd = np.where(np.isfinite(d), d, -1.0)
            where = np.unravel_index(int(np.argmax(dd)), d.shape) if d.ndim else ()
            print(f"{label}:{k}: max diff {m:.3g} at {where}" + (f"  ({nb} entries > {BRANCH_TOL})" if nb else ""))
    return worst, bad, branch


def die(msg: str) -> NoReturn:
    """Print an input error and exit with status 2."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    """Compare the result directories (or files) given on the command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    a, b, tol = Path(args.a), Path(args.b), args.tol
    if not np.isfinite(tol) or tol < 0:
        die("--tol must be finite and nonnegative")
    for p in (a, b):
        if not p.exists():
            die(f"{p} does not exist")
    if a.is_dir() != b.is_dir():
        die("give two directories or two files, not one of each")
    if a.is_dir():
        if a.resolve() == b.resolve():
            die("both arguments are the same directory")
        fa = {p.relative_to(a) for p in a.rglob("*") if p.is_file() and p.suffix in SUFFIXES}
        fb = {p.relative_to(b) for p in b.rglob("*") if p.is_file() and p.suffix in SUFFIXES}
        if not fa and not fb:
            die(f"no {'/'.join(SUFFIXES)} files in either directory; nothing was compared")
        pairs = [(a / r, b / r, str(r)) for r in sorted(fa & fb)]
        missing = [(r, "B") for r in sorted(fa - fb)] + [(r, "A") for r in sorted(fb - fa)]
    else:
        if a.suffix not in SUFFIXES or a.suffix != b.suffix:
            die(f"files must both be one of {', '.join(SUFFIXES)} (got {a.suffix}, {b.suffix})")
        pairs, missing = [(a, b, a.name)], []
    worst, bad, branch = 0.0, 0, 0
    for r, side in missing:
        print(f"MISSING in {side}: {r}"); bad += 1
    for pa, pb, label in pairs:
        w, n, nb = compare_file(pa, pb, label, tol)
        worst, bad, branch = max(worst, w), bad + n, branch + nb
    if not pairs:
        die("no file exists on both sides; nothing was compared")
    print(f"\n{len(pairs)} files compared ({len(missing)} on one side only); largest finite difference {worst:.3g}; "
          f"{bad} items differ (tol {tol:g}); {branch} entries above {BRANCH_TOL}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
