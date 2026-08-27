"""Loss functions for multi-task outline training."""

from .segmentation import PersonSegmentationLoss
from .total import OutlineLoss, OutlineLossWeights

__all__ = ["OutlineLoss", "OutlineLossWeights", "PersonSegmentationLoss"]
