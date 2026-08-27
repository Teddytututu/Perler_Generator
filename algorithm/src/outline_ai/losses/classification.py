"""Reusable binary and categorical segmentation losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def binary_logit_loss(logits: torch.Tensor, target: torch.Tensor, positive_weight: float) -> torch.Tensor:
    pos_weight = logits.new_tensor([positive_weight])
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: tuple[float, ...] | None = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    if not torch.any(target != ignore_index):
        return logits.sum() * 0.0
    weights = logits.new_tensor(class_weights) if class_weights is not None else None
    return F.cross_entropy(logits, target, weight=weights, ignore_index=ignore_index)
