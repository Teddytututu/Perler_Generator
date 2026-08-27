"""Weighted first-pass objective for the stage-1 outline model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .classification import binary_logit_loss, categorical_loss


@dataclass(frozen=True)
class OutlineLossWeights:
    subject: float = 1.0
    outline_binary: float = 2.0
    outline_type: float = 1.0
    direction: float = 0.25
    endpoint: float = 0.25
    grid_outline: float = 2.0
    grid_type: float = 1.0
    grid_direction: float = 0.25
    grid_endpoint: float = 0.25
    side_outline: float = 0.5


class OutlineLoss(nn.Module):
    def __init__(self, weights: OutlineLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or OutlineLossWeights()

    def forward(
        self,
        prediction: dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
        target: dict[str, torch.Tensor | list[str]],
    ) -> dict[str, torch.Tensor]:
        losses = {
            "subject": binary_logit_loss(prediction["subject_logits"], target["subject"], 2.0),
            "outline_binary": binary_logit_loss(
                prediction["outline_binary_logits"], target["outline_binary"], 20.0
            ),
            "outline_type": categorical_loss(
                prediction["outline_type_logits"], target["outline_type"], (0.1, 1.0, 1.0, 1.0)
            ),
            "direction": categorical_loss(prediction["direction_logits"], target["direction"]),
            "endpoint": binary_logit_loss(prediction["endpoint_logits"], target["endpoint"], 50.0),
            "grid_outline": binary_logit_loss(
                prediction["grid_outline_logits"], target["grid_outline"], 10.0
            ),
            "grid_type": categorical_loss(
                prediction["grid_type_logits"], target["grid_type"], (0.1, 1.0, 1.0, 1.0)
            ),
            "grid_direction": categorical_loss(
                prediction["grid_direction_logits"], target["grid_direction"]
            ),
            "grid_endpoint": binary_logit_loss(
                prediction["grid_endpoint_logits"], target["grid_endpoint"], 20.0
            ),
        }
        side_losses = []
        for side_logits in prediction["side_outline_logits"]:
            side_target = F.interpolate(target["outline_binary"], size=side_logits.shape[-2:], mode="nearest")
            side_losses.append(binary_logit_loss(side_logits, side_target, 20.0))
        losses["side_outline"] = torch.stack(side_losses).mean()
        total = sum(getattr(self.weights, name) * value for name, value in losses.items())
        return {"total": total, **losses}
