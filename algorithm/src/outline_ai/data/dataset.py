"""Filesystem contract for stage-1 outline samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REQUIRED_SAMPLE_FILES = (
    "image.png",
    "subject_mask.png",
    "outline_type.png",
    "outline_binary.png",
    "direction.png",
    "endpoint.png",
    "grid_outline_type.png",
    "grid_outline_binary.png",
    "metadata.json",
)


@dataclass(frozen=True)
class OutlineSample:
    root: Path
    image: np.ndarray
    subject_mask: np.ndarray
    outline_type: np.ndarray
    outline_binary: np.ndarray
    direction: np.ndarray
    endpoint: np.ndarray
    grid_outline_type: np.ndarray
    grid_outline_binary: np.ndarray
    metadata: dict[str, Any]

    @property
    def sample_id(self) -> str:
        return str(self.metadata["sampleId"])


def discover_samples(dataset_root: Path) -> list[Path]:
    """Return sample directories in stable order."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "metadata.json").is_file()
    )


def _load_image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert(mode)).copy()


def load_sample(sample_root: Path) -> OutlineSample:
    """Load one standardized sample without applying training transforms."""
    root = Path(sample_root)
    missing = [name for name in REQUIRED_SAMPLE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {root}: {', '.join(missing)}")

    with (root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return OutlineSample(
        root=root,
        image=_load_image(root / "image.png", "RGB"),
        subject_mask=_load_image(root / "subject_mask.png", "L"),
        outline_type=_load_image(root / "outline_type.png", "L"),
        outline_binary=_load_image(root / "outline_binary.png", "L"),
        direction=_load_image(root / "direction.png", "L"),
        endpoint=_load_image(root / "endpoint.png", "L"),
        grid_outline_type=_load_image(root / "grid_outline_type.png", "L"),
        grid_outline_binary=_load_image(root / "grid_outline_binary.png", "L"),
        metadata=metadata,
    )

