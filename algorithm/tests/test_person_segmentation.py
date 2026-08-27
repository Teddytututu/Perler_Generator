from __future__ import annotations

import numpy as np

from outline_ai.postprocess import clean_subject_mask


def test_clean_subject_mask_keeps_largest_component_and_fills_holes() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:24, 5:25] = 1
    mask[10:14, 10:14] = 0
    mask[28:30, 28:30] = 1
    cleaned = clean_subject_mask(mask, closing_radius=1, minimum_area=16)
    assert cleaned[11, 11] == 1
    assert cleaned[28, 28] == 0
    assert cleaned.sum() == 400


def test_clean_subject_mask_rejects_tiny_prediction() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    assert not clean_subject_mask(mask, minimum_area=8).any()
