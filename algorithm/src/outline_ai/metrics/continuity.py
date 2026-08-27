"""Contour continuity measured by the largest matched predicted path."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def contour_continuity(prediction: np.ndarray, target: np.ndarray, tolerance: int = 1) -> float:
    pred = skeletonize(np.asarray(prediction, dtype=bool))
    truth = skeletonize(np.asarray(target, dtype=bool))
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} != {truth.shape}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    target_length = int(truth.sum())
    if target_length == 0:
        return float(not pred.any())

    near_truth = ndimage.binary_dilation(
        truth,
        structure=ndimage.generate_binary_structure(2, 2),
        iterations=tolerance,
    ) if tolerance else truth
    matched = pred & near_truth
    labels, count = ndimage.label(matched, structure=ndimage.generate_binary_structure(2, 2))
    if not count:
        return 0.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return min(1.0, float(sizes.max()) / target_length)

