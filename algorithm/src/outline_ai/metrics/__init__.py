"""Metrics for pixel accuracy, continuity, topology, and grid smoothness."""

from .closure import ClosureResult, evaluate_closure
from .continuity import contour_continuity
from .pixel_metrics import BinaryMetrics, binary_metrics
from .smoothness import SmoothnessResult, smoothness_score

__all__ = [
    "BinaryMetrics",
    "ClosureResult",
    "SmoothnessResult",
    "binary_metrics",
    "contour_continuity",
    "evaluate_closure",
    "smoothness_score",
]

