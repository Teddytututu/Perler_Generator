from __future__ import annotations

import numpy as np

from outline_ai.baselines import CannyBaseline, MaskBoundaryBaseline
from outline_ai.metrics.closure import evaluate_closure


def test_mask_boundary_baseline_returns_closed_grid() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[20:108, 30:98] = 255
    result = MaskBoundaryBaseline().predict(mask)
    assert result.highres_outline.shape == mask.shape
    assert result.grid_outline.shape == (48, 48)
    assert evaluate_closure(result.grid_outline).strict_closed


def test_canny_baseline_detects_contrast_boundary() -> None:
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    image[24:104, 32:96] = 0
    result = CannyBaseline().predict(image)
    assert result.highres_outline.any()
    assert result.grid_outline.shape == (48, 48)

