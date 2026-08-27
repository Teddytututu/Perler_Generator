"""Subject-mask outer-boundary baseline."""

from __future__ import annotations

import numpy as np

from outline_ai.data.rasterize import mask_to_outer_boundary, project_mask_to_grid_boundary

from .common import BaselineOutput


class MaskBoundaryBaseline:
    def __init__(self, grid_size: tuple[int, int] = (48, 48)) -> None:
        self.grid_size = grid_size

    def predict(self, subject_mask: np.ndarray) -> BaselineOutput:
        highres = mask_to_outer_boundary(subject_mask)
        grid = project_mask_to_grid_boundary(subject_mask, self.grid_size)
        return BaselineOutput(
            highres_outline=highres,
            grid_outline=grid,
            diagnostics={"method": "mask_boundary"},
        )

