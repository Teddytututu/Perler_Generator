#!/usr/bin/env python3
"""Validate, crop, pad, and resize portrait images without geometric distortion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ProcessRecord:
    source_path: str
    output_path: str | None
    status: str
    reject_reason: str | None
    source_width: int | None
    source_height: int | None
    oriented_width: int | None
    oriented_height: int | None
    crop_left: int | None
    crop_top: int | None
    crop_right: int | None
    crop_bottom: int | None
    output_width: int | None
    output_height: int | None
    scale: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--min-source-side", type=int, default=48)
    parser.add_argument("--max-aspect-ratio", type=float, default=1.8)
    parser.add_argument(
        "--crop-policy",
        choices=("pad-square", "center-square"),
        default="pad-square",
        help="pad-square preserves all pixels; center-square removes equal excess from the long axis.",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--background", default="ffffff")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def parse_color(value: str) -> tuple[int, int, int]:
    normalized = value.strip().lstrip("#")
    if len(normalized) != 6:
        raise ValueError("background must be a six-digit RGB hex value")
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))


def discover_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def stable_output_name(relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{relative_path.stem}-{digest}.jpg"


def center_square_box(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def add_safe_padding(image: Image.Image, ratio: float, background: tuple[int, int, int]) -> Image.Image:
    border = max(0, round(max(image.size) * ratio))
    if border == 0:
        return image
    return ImageOps.expand(image, border=(border, border, border, border), fill=background)


def pad_to_square(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas = Image.new("RGB", (side, side), background)
    canvas.paste(image, (left, top))
    return canvas


def process_image(
    source: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    background: tuple[int, int, int],
) -> ProcessRecord:
    relative = source.relative_to(input_root)
    base = ProcessRecord(
        source_path=relative.as_posix(),
        output_path=None,
        status="rejected",
        reject_reason=None,
        source_width=None,
        source_height=None,
        oriented_width=None,
        oriented_height=None,
        crop_left=None,
        crop_top=None,
        crop_right=None,
        crop_bottom=None,
        output_width=None,
        output_height=None,
        scale=None,
    )

    try:
        with Image.open(source) as raw:
            base.source_width, base.source_height = raw.size
            image = ImageOps.exif_transpose(raw)
            image.load()
            image = image.convert("RGB")
    except Exception as exc:
        base.reject_reason = f"decode_error:{type(exc).__name__}"
        return base

    width, height = image.size
    base.oriented_width = width
    base.oriented_height = height
    if min(width, height) < args.min_source_side:
        base.reject_reason = "source_too_small"
        return base
    if max(width / height, height / width) > args.max_aspect_ratio:
        base.reject_reason = "extreme_aspect_ratio"
        return base

    if args.crop_policy == "center-square":
        crop_box = center_square_box(width, height)
        image = image.crop(crop_box)
    else:
        crop_box = (0, 0, width, height)

    base.crop_left, base.crop_top, base.crop_right, base.crop_bottom = crop_box
    image = add_safe_padding(image, args.padding_ratio, background)
    image = pad_to_square(image, background)

    pre_resize_side = image.width
    image = image.resize((args.output_size, args.output_size), Image.Resampling.LANCZOS)
    base.scale = round(args.output_size / pre_resize_side, 8)

    output_name = stable_output_name(relative)
    output_path = output_root / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=args.quality, subsampling=0, optimize=True)

    base.output_path = output_path.relative_to(output_root).as_posix()
    base.output_width = args.output_size
    base.output_height = args.output_size
    base.status = "accepted"
    base.reject_reason = None
    return base


def main() -> None:
    args = parse_args()
    if args.output_size <= 0:
        raise SystemExit("output-size must be positive")
    if args.min_source_side <= 0:
        raise SystemExit("min-source-side must be positive")
    if not math.isfinite(args.padding_ratio) or args.padding_ratio < 0:
        raise SystemExit("padding-ratio must be a finite non-negative value")

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise SystemExit("Output directory must not be inside the input directory")

    background = parse_color(args.background)
    paths = discover_images(input_root)
    if args.limit is not None:
        paths = paths[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        process_image(path, input_root, output_root, args, background)
        for path in paths
    ]

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    csv_path = manifest_path.with_suffix(".csv")
    if records:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)

    summary = {
        "total": len(records),
        "accepted": sum(record.status == "accepted" for record in records),
        "rejected": sum(record.status == "rejected" for record in records),
        "output_size": args.output_size,
        "crop_policy": args.crop_policy,
        "padding_ratio": args.padding_ratio,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
