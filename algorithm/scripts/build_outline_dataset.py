#!/usr/bin/env python3
"""Build review-required bootstrap outline samples from image/mask pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.data.rasterize import (  # noqa: E402
    boundary_to_type_map,
    compute_direction_labels,
    mask_to_outer_boundary,
    project_mask_to_grid_boundary,
)
from outline_ai.postprocess.endpoint import endpoint_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def fit_square(image: Image.Image, size: int, resample: Image.Resampling, fill: int | tuple[int, ...]) -> Image.Image:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample)
    mode = resized.mode
    canvas = Image.new(mode, (size, size), fill)
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def save_label(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(path, format="PNG")


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.grid_size <= 0:
        raise SystemExit("image-size and grid-size must be positive")
    if not args.images.is_dir() or not args.masks.is_dir():
        raise SystemExit("images and masks must be existing directories")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in args.images.iterdir() if path.is_file())
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    built = 0
    skipped: list[str] = []
    for image_path in image_paths:
        mask_path = args.masks / image_path.name
        if not mask_path.is_file():
            skipped.append(f"{image_path.name}:missing_mask")
            continue
        sample_id = f"sample_{image_path.stem}"
        sample_dir = output / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as raw_image, Image.open(mask_path) as raw_mask:
            source_width, source_height = raw_image.size
            image = fit_square(ImageOps.exif_transpose(raw_image).convert("RGB"), args.image_size, Image.Resampling.LANCZOS, (255, 255, 255))
            mask = fit_square(ImageOps.exif_transpose(raw_mask).convert("L"), args.image_size, Image.Resampling.NEAREST, 0)
        subject = (np.asarray(mask) > args.mask_threshold).astype(np.uint8)
        boundary = mask_to_outer_boundary(subject)
        outline_type = boundary_to_type_map(boundary)
        direction = compute_direction_labels(boundary)
        endpoints = endpoint_mask(boundary) * 255
        grid_boundary = project_mask_to_grid_boundary(subject, (args.grid_size, args.grid_size))
        grid_type = boundary_to_type_map(grid_boundary)

        image.save(sample_dir / "image.png", format="PNG")
        save_label(sample_dir / "subject_mask.png", subject * 255)
        save_label(sample_dir / "outline_type.png", outline_type)
        save_label(sample_dir / "outline_binary.png", boundary * 255)
        save_label(sample_dir / "direction.png", direction)
        save_label(sample_dir / "endpoint.png", endpoints)
        save_label(sample_dir / "grid_outline_type.png", grid_type)
        save_label(sample_dir / "grid_outline_binary.png", grid_boundary * 255)
        metadata = {
            "sampleId": sample_id,
            "domain": "anime_portrait",
            "sourceWidth": source_width,
            "sourceHeight": source_height,
            "subjectCount": 1,
            "primarySubjectId": 1,
            "isTruncated": bool(subject[0].any() or subject[-1].any() or subject[:, 0].any() or subject[:, -1].any()),
            "isOccluded": False,
            "requiresOuterClosure": True,
            "allowedGapGrid": 1,
            "gridWidth": args.grid_size,
            "gridHeight": args.grid_size,
            "annotationVersion": "outline-bootstrap-v1",
            "reviewStatus": "needs_review",
            "highresLabelSource": "foreground_mask_boundary",
            "gridLabelSource": "automatic_boundary_projection",
            "sourceImage": image_path.name,
        }
        with (sample_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        built += 1

    print(json.dumps({"built": built, "skipped": len(skipped), "skip_reasons": skipped[:50]}, indent=2))


if __name__ == "__main__":
    main()
