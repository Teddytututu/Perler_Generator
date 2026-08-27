"""Traditional and pretrained edge baselines."""

from .canny import CannyBaseline
from .common import BaselineOutput
from .learned_edge import TorchEdgeBaseline
from .mask_boundary import MaskBoundaryBaseline

__all__ = ["BaselineOutput", "CannyBaseline", "MaskBoundaryBaseline", "TorchEdgeBaseline"]

