"""Overlay semantic outline classes on an RGB image."""

from __future__ import annotations

import numpy as np
from PIL import Image


DEFAULT_COLORS = {
    1: (255, 48, 48),
    2: (48, 180, 255),
    3: (255, 210, 32),
    255: (180, 180, 180),
}


def overlay_outline(
    image: Image.Image | np.ndarray,
    outline_type: np.ndarray,
    alpha: float = 0.8,
    colors: dict[int, tuple[int, int, int]] | None = None,
) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    base = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
    labels = np.asarray(outline_type)
    if base.shape[:2] != labels.shape:
        raise ValueError(f"Shape mismatch: image={base.shape[:2]}, labels={labels.shape}")
    result = base.astype(np.float32)
    for class_id, color in (colors or DEFAULT_COLORS).items():
        selected = labels == class_id
        result[selected] = (1.0 - alpha) * result[selected] + alpha * np.asarray(color)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")

