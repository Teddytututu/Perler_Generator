"""End-to-end multi-task outline model with a replaceable encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import MultiScaleOutlineDecoder
from .encoder import TinyHierarchicalEncoder
from .grid_projection import SemanticGridProjection
from .heads import GridPredictionHeads, PredictionHeads


@dataclass(frozen=True)
class OutlineModelConfig:
    encoder_channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    decoder_channels: int = 64
    outline_classes: int = 4
    direction_classes: int = 8
    grid_size: tuple[int, int] = (48, 48)


class OutlineModel(nn.Module):
    def __init__(self, config: OutlineModelConfig | None = None, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config or OutlineModelConfig()
        self.encoder = encoder or TinyHierarchicalEncoder(self.config.encoder_channels)
        self.decoder = MultiScaleOutlineDecoder(self.config.encoder_channels, self.config.decoder_channels)
        self.highres_heads = PredictionHeads(
            self.config.decoder_channels,
            self.config.outline_classes,
            self.config.direction_classes,
        )
        self.grid_projection = SemanticGridProjection(
            self.config.decoder_channels,
            self.config.decoder_channels,
            self.config.grid_size,
        )
        self.grid_heads = GridPredictionHeads(
            self.config.decoder_channels,
            self.config.outline_classes,
            self.config.direction_classes,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], got {tuple(image.shape)}")
        features = self.encoder(image)
        decoder_feature, side_logits = self.decoder(features)
        lowres_outputs = self.highres_heads(decoder_feature)
        highres_outputs = {
            name: F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
            for name, value in lowres_outputs.items()
        }
        grid_feature = self.grid_projection(
            decoder_feature,
            lowres_outputs["subject_logits"],
            lowres_outputs["outline_binary_logits"],
        )
        return {
            **highres_outputs,
            **self.grid_heads(grid_feature),
            "side_outline_logits": side_logits,
        }

