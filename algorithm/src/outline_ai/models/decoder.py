"""HED/RCF-inspired fusion of all hierarchical feature levels."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleOutlineDecoder(nn.Module):
    def __init__(self, encoder_channels: tuple[int, ...], decoder_channels: int = 64) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, kernel_size=1) for channels in encoder_channels
        )
        self.side_heads = nn.ModuleList(
            nn.Conv2d(decoder_channels, 1, kernel_size=1) for _ in encoder_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(encoder_channels), decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        target_size = features[0].shape[-2:]
        projected = []
        side_logits = []
        for feature, lateral, side_head in zip(features, self.lateral, self.side_heads, strict=True):
            value = lateral(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            projected.append(value)
            side_logits.append(side_head(value))
        return self.fuse(torch.cat(projected, dim=1)), tuple(side_logits)

