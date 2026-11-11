"""Compare two complete result directories, e.g. runs on two computers.

Usage: python scripts/10_compare_runs.py RESULTS_A RESULTS_B [--tol 1e-6]

Every numeric array in every .npz/.npy file and every number in every .json file is compared.
Differences above --tol are listed; differences above 0.05 (mm, deg, or normalized units) are flagged as
possible equilibrium-branch changes.  Exit status 1 if any difference exceeds --tol.
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np


def flat_json(x, key=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat_json(v, f"{key}.{k}" if key else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat_json(v, f"{key}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield key, float(x)
    else:
        yield key, x


SKIP_JSON = ("time", "time_s", "time_fit_median_s", "time_knee_median_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tol", type=float, default=1e-6)
    a, b = Path(ap.parse_args().a), Path(ap.parse_args().b)
    tol = ap.parse_args().tol
    worst, bad, branch = 0.0, 0, 0
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in (".npz", ".npy", ".json"))
    for rel in files:
        pa, pb = a / rel, b / rel
        if not pb.exists():
            print(f"MISSING in B: {rel}"); bad += 1; continue
        pairs = []
        if rel.suffix == ".json":
            ja, jb = dict(flat_json(json.loads(pa.read_text()))), dict(flat_json(json.loads(pb.read_text())))
            for k in sorted(set(ja) | set(jb)):
                if k.split(".")[-1].split("[")[0] in SKIP_JSON:
                    continue
                va, vb = ja.get(k), jb.get(k)
                if isinstance(va, float) and isinstance(vb, float):
                    pairs.append((k, np.array([va]), np.array([vb])))
                elif va != vb:
                    print(f"{rel}:{k}: {va!r} != {vb!r}"); bad += 1
        else:
            za = np.load(pa, allow_pickle=False) if rel.suffix == ".npz" else {"": np.load(pa)}
            zb = np.load(pb, allow_pickle=False) if rel.suffix == ".npz" else {"": np.load(pb)}
            for k in (za.files if rel.suffix == ".npz" else [""]):
                xa, xb = np.asarray(za[k]), np.asarray(zb[k])
                if xa.shape != xb.shape:
                    print(f"{rel}:{k}: shape {xa.shape} != {xb.shape}"); bad += 1; continue
                if xa.dtype.kind in "fc":
                    pairs.append((k, xa, xb))
                elif not np.array_equal(xa, xb):
                    print(f"{rel}:{k}: values differ"); bad += 1
        for k, xa, xb in pairs:
            d = np.abs(xa.astype(float) - xb.astype(float))
            d = np.where(np.isnan(xa) & np.isnan(xb), 0.0, d)
            m = float(np.nanmax(d)) if d.size else 0.0
            worst = max(worst, m)
            if m > tol:
                bad += 1
                nb = int((d > 0.05).sum()); branch += nb
                where = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.ndim else ()
                print(f"{rel}:{k}: max diff {m:.3g} at {where}" + (f"  ({nb} entries > 0.05)" if nb else ""))
    print(f"\n{len(files)} files compared; largest difference {worst:.3g}; "
          f"{bad} items above tol {tol:g}; {branch} entries above 0.05")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
