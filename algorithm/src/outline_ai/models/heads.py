"""High-resolution and grid-level prediction heads."""

from __future__ import annotations

import torch
from torch import nn


class PredictionHeads(nn.Module):
    def __init__(self, channels: int, outline_classes: int = 4, direction_classes: int = 8) -> None:
        super().__init__()
        self.subject = nn.Conv2d(channels, 1, 1)
        self.outline_binary = nn.Conv2d(channels, 1, 1)
        self.outline_type = nn.Conv2d(channels, outline_classes, 1)
        self.direction = nn.Conv2d(channels, direction_classes, 1)
        self.endpoint = nn.Conv2d(channels, 1, 1)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "subject_logits": self.subject(feature),
            "outline_binary_logits": self.outline_binary(feature),
            "outline_type_logits": self.outline_type(feature),
            "direction_logits": self.direction(feature),
            "endpoint_logits": self.endpoint(feature),
        }


class GridPredictionHeads(nn.Module):
    def __init__(self, channels: int, outline_classes: int = 4, direction_classes: int = 8) -> None:
        super().__init__()
        self.outline = nn.Conv2d(channels, 1, 1)
        self.outline_type = nn.Conv2d(channels, outline_classes, 1)
        self.direction = nn.Conv2d(channels, direction_classes, 1)
        self.endpoint = nn.Conv2d(channels, 1, 1)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "grid_outline_logits": self.outline(feature),
            "grid_type_logits": self.outline_type(feature),
            "grid_direction_logits": self.direction(feature),
            "grid_endpoint_logits": self.endpoint(feature),
        }

