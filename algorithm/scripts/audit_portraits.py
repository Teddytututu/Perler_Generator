#!/usr/bin/env python3
"""Audit an image directory and write reproducible quality statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from PIL import Image, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageRecord:
    path: str
    width: int | None
    height: int | None
    mode: str | None
    format: str | None
    file_bytes: int
    aspect_ratio: float | None
    status: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def discover_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def inspect_image(path: Path, root: Path) -> ImageRecord:
    try:
        with Image.open(path) as image:
            source_format = image.format
            image = ImageOps.exif_transpose(image)
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("non-positive image dimensions")
            return ImageRecord(
                path=path.relative_to(root).as_posix(),
                width=width,
                height=height,
                mode=image.mode,
                format=source_format,
                file_bytes=path.stat().st_size,
                aspect_ratio=round(width / height, 6),
                status="valid",
                error=None,
            )
    except Exception as exc:  # The report must retain every unreadable file.
        return ImageRecord(
            path=path.relative_to(root).as_posix(),
            width=None,
            height=None,
            mode=None,
            format=None,
            file_bytes=path.stat().st_size,
            aspect_ratio=None,
            status="invalid",
            error=f"{type(exc).__name__}: {exc}",
        )


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    index = round((len(values) - 1) * fraction)
    return sorted(values)[index]


def build_summary(records: list[ImageRecord]) -> dict[str, object]:
    valid = [record for record in records if record.status == "valid"]
    widths = [record.width for record in valid if record.width is not None]
    heights = [record.height for record in valid if record.height is not None]
    min_sides = [min(record.width, record.height) for record in valid if record.width and record.height]
    dimension_counts = Counter(f"{record.width}x{record.height}" for record in valid)
    mode_counts = Counter(record.mode for record in valid)
    format_counts = Counter(record.format for record in valid)

    return {
        "total_files": len(records),
        "valid_files": len(valid),
        "invalid_files": len(records) - len(valid),
        "total_bytes": sum(record.file_bytes for record in records),
        "width": {
            "min": min(widths, default=None),
            "median": median(widths) if widths else None,
            "p90": percentile(widths, 0.90),
            "max": max(widths, default=None),
        },
        "height": {
            "min": min(heights, default=None),
            "median": median(heights) if heights else None,
            "p90": percentile(heights, 0.90),
            "max": max(heights, default=None),
        },
        "minimum_side_buckets": {
            "lt_48": sum(value < 48 for value in min_sides),
            "48_to_63": sum(48 <= value < 64 for value in min_sides),
            "64_to_95": sum(64 <= value < 96 for value in min_sides),
            "96_to_127": sum(96 <= value < 128 for value in min_sides),
            "128_to_255": sum(128 <= value < 256 for value in min_sides),
            "gte_256": sum(value >= 256 for value in min_sides),
        },
        "top_dimensions": dimension_counts.most_common(30),
        "modes": dict(mode_counts),
        "formats": dict(format_counts),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_images(input_dir, args.recursive)
    records = [inspect_image(path, input_dir) for path in paths]
    summary = build_summary(records)

    with (output_dir / "audit_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)

    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
