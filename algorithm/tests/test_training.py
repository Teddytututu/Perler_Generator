from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from outline_ai.losses import OutlineLoss
from outline_ai.models import OutlineModel, OutlineModelConfig


def test_outline_loss_supports_backward() -> None:
    model = OutlineModel(OutlineModelConfig(encoder_channels=(8, 16, 32, 64), decoder_channels=16))
    image = torch.randn(1, 3, 64, 64)
    prediction = model(image)
    target = {
        "subject": torch.zeros(1, 1, 64, 64),
        "outline_binary": torch.zeros(1, 1, 64, 64),
        "outline_type": torch.zeros(1, 64, 64, dtype=torch.long),
        "direction": torch.full((1, 64, 64), 255, dtype=torch.long),
        "endpoint": torch.zeros(1, 1, 64, 64),
        "grid_outline": torch.zeros(1, 1, 48, 48),
        "grid_type": torch.zeros(1, 48, 48, dtype=torch.long),
        "grid_direction": torch.full((1, 48, 48), 255, dtype=torch.long),
        "grid_endpoint": torch.zeros(1, 1, 48, 48),
    }
    losses = OutlineLoss()(prediction, target)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
