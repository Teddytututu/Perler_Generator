from __future__ import annotations

import numpy as np

from outline_ai.metrics.continuity import contour_continuity
from outline_ai.metrics.pixel_metrics import binary_metrics
from outline_ai.metrics.smoothness import smoothness_score


def test_binary_metrics_exact_match() -> None:
    value = np.eye(8, dtype=np.uint8)
    result = binary_metrics(value, value)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_binary_metrics_tolerance_accepts_one_pixel_offset() -> None:
    target = np.zeros((12, 12), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[5, 2:10] = 1
    prediction[6, 2:10] = 1
    assert binary_metrics(prediction, target, tolerance=0).f1 == 0.0
    assert binary_metrics(prediction, target, tolerance=1).f1 == 1.0


def test_continuity_drops_when_path_is_fragmented() -> None:
    target = np.zeros((12, 12), dtype=np.uint8)
    target[5, 2:10] = 1
    prediction = target.copy()
    prediction[5, 6] = 0
    assert contour_continuity(prediction, target, tolerance=0) < 1.0


def test_smoothness_penalizes_isolated_pixels() -> None:
    clean = np.zeros((12, 12), dtype=np.uint8)
    clean[6, 2:10] = 1
    noisy = clean.copy()
    noisy[1, 1] = 1
    assert smoothness_score(noisy).score < smoothness_score(clean).score

