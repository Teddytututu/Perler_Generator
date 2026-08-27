"""Endpoint detection on one-pixel, eight-connected outlines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def endpoint_mask(outline: np.ndarray) -> np.ndarray:
    binary = np.asarray(outline, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D outline, got shape {binary.shape}")
    neighbor_count = ndimage.convolve(
        binary.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    ) - binary
    return (binary & (neighbor_count == 1)).astype(np.uint8)


def endpoint_coordinates(outline: np.ndarray) -> np.ndarray:
    return np.argwhere(endpoint_mask(outline) > 0)

