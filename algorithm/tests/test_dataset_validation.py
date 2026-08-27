from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from outline_ai.data.validators import validate_sample_dir


def save_l(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8), mode="L").save(path)


def test_validator_accepts_complete_closed_sample() -> None:
    runtime_root = Path(__file__).parent / "_runtime_validation"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    sample = runtime_root / "sample_000001"
    sample.mkdir(parents=True)
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    subject = np.zeros((32, 32), dtype=np.uint8)
    subject[4:28, 4:28] = 255
    outline = np.zeros((32, 32), dtype=np.uint8)
    outline[4, 4:28] = 255
    outline[27, 4:28] = 255
    outline[4:28, 4] = 255
    outline[4:28, 27] = 255
    outline_type = (outline > 0).astype(np.uint8)
    grid = np.zeros((48, 48), dtype=np.uint8)
    grid[5, 5:43] = 255
    grid[42, 5:43] = 255
    grid[5:43, 5] = 255
    grid[5:43, 42] = 255
    grid_type = (grid > 0).astype(np.uint8)

    Image.fromarray(image, mode="RGB").save(sample / "image.png")
    save_l(sample / "subject_mask.png", subject)
    save_l(sample / "outline_binary.png", outline)
    save_l(sample / "outline_type.png", outline_type)
    save_l(sample / "direction.png", np.full((32, 32), 255, dtype=np.uint8))
    save_l(sample / "endpoint.png", np.zeros((32, 32), dtype=np.uint8))
    save_l(sample / "grid_outline_binary.png", grid)
    save_l(sample / "grid_outline_type.png", grid_type)
    metadata = {
        "sampleId": sample.name,
        "domain": "anime_portrait",
        "sourceWidth": 32,
        "sourceHeight": 32,
        "subjectCount": 1,
        "primarySubjectId": 1,
        "isTruncated": False,
        "isOccluded": False,
        "requiresOuterClosure": True,
        "allowedGapGrid": 1,
        "gridWidth": 48,
        "gridHeight": 48,
        "annotationVersion": "outline-v1",
        "reviewStatus": "approved",
    }
    (sample / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    try:
        result = validate_sample_dir(sample)
        assert result.valid, result.issues
    finally:
        shutil.rmtree(runtime_root)
