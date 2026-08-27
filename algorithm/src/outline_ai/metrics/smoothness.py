"""Deterministic local smoothness proxies for discrete outlines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class SmoothnessResult:
    score: float
    direction_reversal_ratio: float
    spike_ratio: float
    thickness_change_ratio: float
    isolated_pixel_ratio: float


def smoothness_score(outline: np.ndarray) -> SmoothnessResult:
    original = np.asarray(outline, dtype=bool)
    skeleton = skeletonize(original)
    count = int(skeleton.sum())
    if not count:
        return SmoothnessResult(1.0, 0.0, 0.0, 0.0, 0.0)

    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant") - skeleton
    isolated = int((skeleton & (neighbors == 0)).sum())
    endpoints = int((skeleton & (neighbors == 1)).sum())
    branches = int((skeleton & (neighbors > 2)).sum())

    # Branches and excessive endpoints are stable proxies for local reversals
    # and one-cell spikes without requiring global path ordering.
    reversal_ratio = branches / count
    spike_ratio = max(0, endpoints - 2) / count
    removed_by_skeleton = max(0, int(original.sum()) - count)
    thickness_ratio = removed_by_skeleton / max(1, int(original.sum()))
    isolated_ratio = isolated / count
    score = 1.0 - (0.35 * reversal_ratio + 0.35 * spike_ratio + 0.2 * thickness_ratio + 0.1 * isolated_ratio)
    return SmoothnessResult(
        score=float(np.clip(score, 0.0, 1.0)),
        direction_reversal_ratio=reversal_ratio,
        spike_ratio=spike_ratio,
        thickness_change_ratio=thickness_ratio,
        isolated_pixel_ratio=isolated_ratio,
    )

