"""Canny, morphology, skeletonization, and grid-projection baseline."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import skeletonize

from outline_ai.data.rasterize import project_boundary_to_grid

from .common import BaselineOutput


class CannyBaseline:
    def __init__(
        self,
        sigma: float = 1.4,
        low_threshold: float = 0.08,
        high_threshold: float = 0.20,
        closing_iterations: int = 1,
        grid_size: tuple[int, int] = (48, 48),
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if not 0 <= low_threshold < high_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.closing_iterations = closing_iterations
        self.grid_size = grid_size

    def predict(self, image: np.ndarray) -> BaselineOutput:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError(f"Expected an RGB image, got shape {value.shape}")
        gray = rgb2gray(value[..., :3])
        edges = canny(
            gray,
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
        if self.closing_iterations:
            edges = ndimage.binary_closing(
                edges,
                structure=ndimage.generate_binary_structure(2, 2),
                iterations=self.closing_iterations,
            )
        highres = skeletonize(edges).astype(np.uint8)
        grid = project_boundary_to_grid(highres, self.grid_size)
        return BaselineOutput(
            highres_outline=highres,
            grid_outline=grid,
            diagnostics={
                "method": "canny",
                "sigma": self.sigma,
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
            },
        )

