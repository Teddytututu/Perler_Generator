"""Strict and geometry-only repairable closure metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from outline_ai.postprocess.endpoint import endpoint_coordinates


@dataclass(frozen=True)
class ClosureResult:
    strict_closed: bool
    repairable_closed: bool
    component_count: int
    endpoint_count: int
    enclosed_area: int
    smallest_gap: int | None


def _enclosed_area(outline: np.ndarray) -> int:
    filled = ndimage.binary_fill_holes(outline)
    return int((filled & ~outline).sum())


def evaluate_closure(outline: np.ndarray, allowed_gap: int = 1) -> ClosureResult:
    """Evaluate one outer-contour raster.

    Repairability here is intentionally geometry-only. The v0.5 gap-repair
    module must additionally check direction, semantic ownership, and path cost.
    """
    if allowed_gap < 0:
        raise ValueError("allowed_gap must be non-negative")
    binary = skeletonize(np.asarray(outline, dtype=bool))
    structure = ndimage.generate_binary_structure(2, 2)
    _, component_count = ndimage.label(binary, structure=structure)
    endpoints = endpoint_coordinates(binary)
    enclosed_area = _enclosed_area(binary)
    strict = bool(binary.any() and component_count == 1 and len(endpoints) == 0 and enclosed_area > 0)

    smallest_gap: int | None = None
    repairable = strict
    if not strict and component_count == 1 and len(endpoints) == 2:
        delta = np.abs(endpoints[0] - endpoints[1])
        smallest_gap = max(0, int(delta.max()) - 1)
        repairable = smallest_gap <= allowed_gap

    return ClosureResult(
        strict_closed=strict,
        repairable_closed=repairable,
        component_count=int(component_count),
        endpoint_count=len(endpoints),
        enclosed_area=enclosed_area,
        smallest_gap=smallest_gap,
    )

