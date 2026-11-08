"""Measurement subsets compared in the identifiability analysis, and measurement noise model."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from .protocols import feature_names, TEST_UNITS
from .sampling import NOISE

TEST_NOISE_SEED = 2024   # one fixed noise realization shared by all evaluations on the test knees

FEATURES = feature_names()
_TESTS = {"gap": ("gapM", "gapL"), "VV": ("var", "val"), "AP": ("ant", "post"), "IE": ("int", "ext")}

# name: (test families, flexion angles)
SUBSETS = {
    "Gaps 0/90": (("gap",), (0, 90)),
    "Gaps": (("gap",), (0, 30, 60, 90)),
    "VV": (("VV",), (0, 30, 60, 90)),
    "Gaps+VV": (("gap", "VV"), (0, 30, 60, 90)),
    "Gaps+VV+AP": (("gap", "VV", "AP"), (0, 30, 60, 90)),
    "Gaps+VV+IE": (("gap", "VV", "IE"), (0, 30, 60, 90)),
    "All 0": (("gap", "VV", "AP", "IE"), (0,)),
    "All 0/90": (("gap", "VV", "AP", "IE"), (0, 90)),
    "All": (("gap", "VV", "AP", "IE"), (0, 30, 60, 90)),
}


def subset_index(name: str) -> np.ndarray:
    """Indices of the examination outputs that belong to measurement subset `name`."""
    fams, flex = SUBSETS[name]
    tests = [t for f in fams for t in _TESTS[f]]
    return np.array([i for i, n in enumerate(FEATURES)
                     if n.split("@")[0] in tests and int(n.split("@")[1]) in flex])


def noise_sd(names: Sequence[str] = FEATURES) -> np.ndarray:
    """Measurement-noise standard deviation of each output (mm or deg)."""
    return np.array([NOISE[TEST_UNITS[n.split("@")[0]]] for n in names])


def test_noise(shape: tuple, scale: float = 1.0) -> np.ndarray:
    """Fixed measurement-noise realization for the test knees (same seed in every script)."""
    return np.random.default_rng(TEST_NOISE_SEED).normal(size=shape) * noise_sd() * scale
