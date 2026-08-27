"""Neural model components for semantic outline generation."""

from .outline_model import OutlineModel, OutlineModelConfig
from .person_segmenter import PretrainedPersonSegmenter

__all__ = ["OutlineModel", "OutlineModelConfig", "PretrainedPersonSegmenter"]
