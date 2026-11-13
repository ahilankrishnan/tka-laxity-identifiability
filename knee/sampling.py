"""Prior over ligament parameters, nuisance geometry, and measurement noise."""
from __future__ import annotations

from typing import Optional

import numpy as np
from .model import GROUPS, BUNDLES, Knee

# Uniform priors on log10 stiffness (N per unit strain, per bundle) and reference strain.
PRIOR = {
    "k_MCL": (1500.0, 5500.0), "eps_MCL": (0.00, 0.08),
    "k_LCL": (1000.0, 4000.0), "eps_LCL": (0.00, 0.08),
    "k_PCL": (2500.0, 9000.0), "eps_PCL": (-0.04, 0.04),
    "k_CAP": (500.0, 3000.0), "eps_CAP": (-0.04, 0.04),
}
PARAM_NAMES = [f"{p}_{g}" for g in GROUPS for p in ("k", "eps")]

# Measurement noise (1 SD): navigation-grade tracking accuracy.
NOISE = {"mm": 0.5, "deg": 0.5}
ATTACH_SD = 1.5   # mm, per coordinate, nuisance attachment variability

# Seeds: characterization knee i uses seed i, test knee i uses TEST_SEED0 + i; the attachment offsets of a
# knee use seed + OFFSET_SEED_SHIFT, so that they are drawn independently of its parameters.
TEST_SEED0 = 100_000
OFFSET_SEED_SHIFT = 7_000_000


def sample_params(rng: np.random.Generator, n: int) -> list[dict]:
    """Draw n parameter sets: stiffness log-uniform, reference strain uniform within PRIOR."""
    out = []
    for _ in range(n):
        p = {}
        for name in PARAM_NAMES:
            lo, hi = PRIOR[name]
            p[name] = 10 ** rng.uniform(np.log10(lo), np.log10(hi)) if name.startswith("k_") else rng.uniform(lo, hi)
        out.append(p)
    return out


def sample_offsets(rng: np.random.Generator, sd: float = ATTACH_SD) -> Optional[np.ndarray]:
    """Gaussian perturbations of all femoral and tibial attachment coordinates, mm (None if sd = 0)."""
    return rng.normal(0.0, sd, size=(len(BUNDLES), 2, 3)) if sd > 0 else None


def to_vector(p: dict) -> np.ndarray:
    """Normalized parameter vector in [0, 1]^8 (log scale for stiffness)."""
    v = []
    for name in PARAM_NAMES:
        lo, hi = PRIOR[name]
        v.append((np.log10(p[name]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if name.startswith("k_")
                 else (p[name] - lo) / (hi - lo))
    return np.array(v)


def from_vector(v: np.ndarray) -> dict:
    """Physical parameter dictionary from a normalized vector (inverse of `to_vector`)."""
    p = {}
    for x, name in zip(v, PARAM_NAMES):
        lo, hi = PRIOR[name]
        p[name] = 10 ** (np.log10(lo) + x * (np.log10(hi) - np.log10(lo))) if name.startswith("k_") else lo + x * (hi - lo)
    return p


def knee_from_seed(seed: int, attach_sd: float = ATTACH_SD, **kw) -> tuple[Knee, dict]:
    """Virtual knee with parameters and attachment offsets drawn from independent seeds (see TEST_SEED0)."""
    p = sample_params(np.random.default_rng(seed), 1)[0]
    offsets = sample_offsets(np.random.default_rng(seed + OFFSET_SEED_SHIFT), attach_sd)
    return Knee(params=p, attach_offsets=offsets, **kw), p
