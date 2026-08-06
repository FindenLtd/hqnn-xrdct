from __future__ import annotations

import numpy as np


def maxpool_1d(pattern: np.ndarray, bins: int) -> np.ndarray:
    chunks = np.array_split(np.arange(pattern.size), bins)
    return np.asarray([np.max(pattern[index]) for index in chunks], dtype=np.float64)


def l2_normalize(patterns: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(patterns, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)

