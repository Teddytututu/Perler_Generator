"""Learned semantic projection from decoder features to the bead grid."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SemanticGridProjection(nn.Module):
    def __init__(self, feature_channels: int, output_channels: int, grid_size: tuple[int, int] = (48, 48)) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.refine = nn.Sequential(
            nn.Conv2d(feature_channels + 2, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(
        self,
        decoder_feature: torch.Tensor,
        subject_logits: torch.Tensor,
        outline_logits: torch.Tensor,
    ) -> torch.Tensor:
        feature = F.adaptive_avg_pool2d(decoder_feature, self.grid_size)
        subject = F.adaptive_avg_pool2d(torch.sigmoid(subject_logits), self.grid_size)
        outline = F.adaptive_max_pool2d(torch.sigmoid(outline_logits), self.grid_size)
        return self.refine(torch.cat((feature, subject, outline), dim=1))

