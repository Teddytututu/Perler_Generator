"""Adapter for evaluating a pretrained PyTorch edge detector as a baseline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize

from outline_ai.data.rasterize import project_boundary_to_grid

from .common import BaselineOutput


class TorchEdgeBaseline:
    """Wrap HED, PiDiNet, or another model behind the baseline contract."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        threshold: float = 0.5,
        grid_size: tuple[int, int] = (48, 48),
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.threshold = threshold
        self.grid_size = grid_size
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> BaselineOutput:
        tensor = self.preprocess(np.asarray(image)).to(self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        output = self.model(tensor)
        if isinstance(output, dict):
            output = output.get("outline_logits", output.get("logits"))
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if not isinstance(output, torch.Tensor):
            raise TypeError("Learned edge model must return a tensor, sequence, or logits dictionary")
        probability = torch.sigmoid(output)[0, 0].detach().cpu().numpy()
        if probability.shape != image.shape[:2]:
            resized = Image.fromarray(probability.astype(np.float32), mode="F").resize(
                (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
            )
            probability = np.asarray(resized)
        highres = skeletonize(probability >= self.threshold).astype(np.uint8)
        grid = project_boundary_to_grid(highres, self.grid_size)
        return BaselineOutput(
            highres_outline=highres,
            grid_outline=grid,
            diagnostics={"method": "learned_edge", "threshold": self.threshold},
        )

