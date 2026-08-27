from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from outline_ai.models import OutlineModel, OutlineModelConfig


def test_outline_model_output_contract() -> None:
    config = OutlineModelConfig(
        encoder_channels=(8, 16, 32, 64),
        decoder_channels=16,
        grid_size=(48, 48),
    )
    model = OutlineModel(config).eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        output = model(image)

    assert output["subject_logits"].shape == (1, 1, 128, 128)
    assert output["outline_binary_logits"].shape == (1, 1, 128, 128)
    assert output["outline_type_logits"].shape == (1, 4, 128, 128)
    assert output["direction_logits"].shape == (1, 8, 128, 128)
    assert output["endpoint_logits"].shape == (1, 1, 128, 128)
    assert output["grid_outline_logits"].shape == (1, 1, 48, 48)
    assert output["grid_type_logits"].shape == (1, 4, 48, 48)
    assert output["grid_direction_logits"].shape == (1, 8, 48, 48)
    assert output["grid_endpoint_logits"].shape == (1, 1, 48, 48)
    assert len(output["side_outline_logits"]) == 4


def test_outline_model_rejects_invalid_input() -> None:
    model = OutlineModel(OutlineModelConfig(encoder_channels=(8, 16, 32, 64), decoder_channels=16))
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(1, 1, 64, 64))

