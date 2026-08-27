"""Conservative cleanup for predicted person masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def clean_subject_mask(mask: np.ndarray, closing_radius: int = 2, minimum_area: int = 64) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary.shape}")
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.uint8)
    structure = ndimage.generate_binary_structure(2, 2)
    labels, count = ndimage.label(binary, structure=structure)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax()) if count else 0
    if largest == 0 or sizes[largest] < minimum_area:
        return np.zeros(binary.shape, dtype=np.uint8)
    cleaned = labels == largest
    if closing_radius > 0:
        kernel = ndimage.iterate_structure(structure, closing_radius)
        cleaned = ndimage.binary_closing(cleaned, structure=kernel)
    cleaned = ndimage.binary_fill_holes(cleaned)
    return cleaned.astype(np.uint8)
