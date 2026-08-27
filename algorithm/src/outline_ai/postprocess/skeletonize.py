"""One-pixel outline skeletonization."""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def skeletonize_outline(outline: np.ndarray) -> np.ndarray:
    value = np.asarray(outline)
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D outline, got shape {value.shape}")
    return skeletonize(value > 0).astype(np.uint8)

