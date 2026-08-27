"""Pretrained person segmentation adapted to a binary foreground logit."""

from __future__ import annotations

import torch
from torch import nn


class PretrainedPersonSegmenter(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models.segmentation import (
                DeepLabV3_MobileNet_V3_Large_Weights,
                deeplabv3_mobilenet_v3_large,
            )
        except ImportError as exc:
            raise ImportError("Install the 'model' optional dependencies to use person segmentation") from exc

        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.network = deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=pretrained)
        categories = weights.meta["categories"] if weights is not None else None
        self.person_index = categories.index("person") if categories is not None else 15
        self._backbone_frozen = False

    def freeze_backbone(self) -> None:
        self.network.backbone.requires_grad_(False)
        self._backbone_frozen = True
        self.network.backbone.eval()

    def unfreeze_backbone(self) -> None:
        self.network.backbone.requires_grad_(True)
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            self.network.backbone.eval()
        return self

    def _person_logit(self, class_logits: torch.Tensor) -> torch.Tensor:
        person = class_logits[:, self.person_index : self.person_index + 1]
        before = class_logits[:, : self.person_index]
        after = class_logits[:, self.person_index + 1 :]
        other = torch.logsumexp(torch.cat((before, after), dim=1), dim=1, keepdim=True)
        return person - other

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.network(image)
        result = {"subject_logits": self._person_logit(outputs["out"])}
        if "aux" in outputs:
            result["aux_subject_logits"] = self._person_logit(outputs["aux"])
        return result
