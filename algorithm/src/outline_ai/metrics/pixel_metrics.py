"""Tolerance-aware binary contour metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    true_positive_pred: int
    true_positive_target: int
    predicted_pixels: int
    target_pixels: int


def binary_metrics(prediction: np.ndarray, target: np.ndarray, tolerance: int = 0) -> BinaryMetrics:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} != {truth.shape}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    if tolerance:
        structure = ndimage.generate_binary_structure(2, 2)
        near_truth = ndimage.binary_dilation(truth, structure=structure, iterations=tolerance)
        near_pred = ndimage.binary_dilation(pred, structure=structure, iterations=tolerance)
    else:
        near_truth = truth
        near_pred = pred

    tp_pred = int((pred & near_truth).sum())
    tp_truth = int((truth & near_pred).sum())
    pred_count = int(pred.sum())
    truth_count = int(truth.sum())
    precision = tp_pred / pred_count if pred_count else float(not truth_count)
    recall = tp_truth / truth_count if truth_count else float(not pred_count)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(precision, recall, f1, tp_pred, tp_truth, pred_count, truth_count)

