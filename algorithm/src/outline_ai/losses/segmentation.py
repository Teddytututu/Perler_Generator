"""Binary person segmentation objective and metrics."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PersonSegmentationLoss(nn.Module):
    def __init__(self, positive_weight: float = 2.0, dice_weight: float = 1.0, aux_weight: float = 0.4) -> None:
        super().__init__()
        self.positive_weight = positive_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight

    def _primary(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=logits.new_tensor([self.positive_weight])
        )
        probability = logits.sigmoid()
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce, dice

    def forward(self, prediction: dict[str, torch.Tensor], target: dict) -> dict[str, torch.Tensor]:
        bce, dice = self._primary(prediction["subject_logits"], target["subject"])
        total = bce + self.dice_weight * dice
        result = {"total": total, "bce": bce, "dice_loss": dice}
        if "aux_subject_logits" in prediction:
            aux_bce, aux_dice = self._primary(prediction["aux_subject_logits"], target["subject"])
            aux = aux_bce + self.dice_weight * aux_dice
            result["aux"] = aux
            result["total"] = total + self.aux_weight * aux
        with torch.no_grad():
            predicted = prediction["subject_logits"] >= 0
            expected = target["subject"] > 0.5
            intersection = (predicted & expected).sum().float()
            union = (predicted | expected).sum().float()
            result["iou"] = (intersection + 1.0) / (union + 1.0)
        return result
