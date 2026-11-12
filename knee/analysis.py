"""Shared post-processing: linearized (Fisher) posterior, identifiability index, and best least-squares fits."""
from __future__ import annotations

import numpy as np
from .subsets import noise_sd

# Precision (inverse variance) of a uniform prior on [0, 1]: Var[U(0, 1)] = 1/12.
PRIOR_PRECISION = 12.0


def posterior(J: np.ndarray, idx: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Fisher information F and linearized posterior covariance C for the outputs `idx`.

    J is the sensitivity matrix of all outputs to the normalized parameters; the measurement-noise standard
    deviations are multiplied by `scale`."""
    S = noise_sd()[idx] * scale
    A = J[idx] / S[:, None]
    F = A.T @ A
    C = np.linalg.inv(F + PRIOR_PRECISION * np.eye(J.shape[1]))
    return F, C


def ident_index(C: np.ndarray) -> np.ndarray:
    """Identifiability index 1 - sd_post / sd_prior of each parameter from the posterior covariance C."""
    return 1 - np.sqrt(np.diag(C)) / np.sqrt(1 / PRIOR_PRECISION)


def median_index(Js: np.ndarray, idx: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Across-knee median identifiability index of each parameter for the outputs `idx`."""
    return np.median([ident_index(posterior(J, idx, scale)[1]) for J in Js], 0)


def best_fits(fits: list[dict]) -> dict[int, dict]:
    """Lowest re-evaluated fit error per knee among starts whose endpoint converged."""
    best: dict[int, dict] = {}
    for f in fits:
        if not f["ok_end"]:
            continue
        if f["i"] not in best or f["cost"] < best[f["i"]]["cost"]:
            best[f["i"]] = f
    return best
