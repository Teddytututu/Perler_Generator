from __future__ import annotations

import numpy as np

from outline_ai.data.rasterize import (
    mask_to_outer_boundary,
    project_boundary_to_grid,
    project_mask_to_grid_boundary,
)
from outline_ai.metrics.closure import evaluate_closure


def test_mask_to_outer_boundary_produces_closed_single_pixel_contour() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[6:26, 8:24] = 1
    boundary = mask_to_outer_boundary(mask)

    result = evaluate_closure(boundary)
    assert result.strict_closed
    assert result.endpoint_count == 0
    assert boundary.sum() < mask.sum()


def test_grid_projection_has_requested_shape() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:56, 12:52] = 1
    boundary = mask_to_outer_boundary(mask)
    grid = project_boundary_to_grid(boundary, (48, 48))
    assert grid.shape == (48, 48)
    assert grid.any()


def test_mask_grid_projection_preserves_closed_outer_contour() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[16:112, 24:104] = 1
    grid = project_mask_to_grid_boundary(mask, (48, 48))
    assert evaluate_closure(grid).strict_closed
