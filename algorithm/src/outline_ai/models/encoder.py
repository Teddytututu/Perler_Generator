"""Hierarchical encoder interface and dependency-free bootstrap encoder."""

from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class TinyHierarchicalEncoder(nn.Module):
    """Produces the same four-scale contract expected from SegFormer."""

    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.channels = channels
        self.stem = nn.Sequential(ConvNormAct(3, c1 // 2, stride=2), ConvNormAct(c1 // 2, c1, stride=2))
        self.stage2 = nn.Sequential(ConvNormAct(c1, c2, stride=2), ConvNormAct(c2, c2))
        self.stage3 = nn.Sequential(ConvNormAct(c2, c3, stride=2), ConvNormAct(c3, c3))
        self.stage4 = nn.Sequential(ConvNormAct(c3, c4, stride=2), ConvNormAct(c4, c4))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f1 = self.stem(image)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4

