#!/usr/bin/env python3
"""Render original, mask, high-resolution outline, and grid-outline previews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.data.dataset import discover_samples, load_sample  # noqa: E402
from outline_ai.visualization.overlay import overlay_outline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=256)
    return parser.parse_args()


def fit(image: Image.Image, size: int, resample: Image.Resampling) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((size, size), resample)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(result, ((size - result.width) // 2, (size - result.height) // 2))
    return canvas


def mask_preview(mask: np.ndarray) -> Image.Image:
    binary = np.asarray(mask) > 0
    value = np.zeros((*binary.shape, 3), dtype=np.uint8)
    value[~binary] = (245, 245, 245)
    value[binary] = (45, 52, 64)
    return Image.fromarray(value, mode="RGB")


def grid_preview(grid_type: np.ndarray, cell_size: int) -> Image.Image:
    labels = np.asarray(grid_type)
    height, width = labels.shape
    scale = max(1, cell_size // max(height, width))
    colors = np.full((height, width, 3), 250, dtype=np.uint8)
    colors[labels == 1] = (225, 38, 48)
    colors[labels == 2] = (25, 145, 220)
    colors[labels == 3] = (245, 185, 25)
    image = Image.fromarray(colors, mode="RGB").resize(
        (width * scale, height * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width + 1, scale):
        draw.line((x, 0, x, image.height), fill=(220, 220, 220), width=1)
    for y in range(0, image.height + 1, scale):
        draw.line((0, y, image.width, y), fill=(220, 220, 220), width=1)
    return fit(image, cell_size, Image.Resampling.NEAREST)


def main() -> None:
    args = parse_args()
    paths = discover_samples(args.dataset)[: args.limit]
    if not paths:
        raise SystemExit("No standardized samples found")

    headers = ("Original", "Subject mask", "512 outline overlay", "48 x 48 grid")
    label_height = 30
    row_height = args.cell_size + label_height
    sheet = Image.new("RGB", (args.cell_size * 4, row_height * (len(paths) + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    for column, header in enumerate(headers):
        draw.text((column * args.cell_size + 8, 8), header, fill="black")

    for row, path in enumerate(paths, start=1):
        sample = load_sample(path)
        original = Image.fromarray(sample.image, mode="RGB")
        panels = (
            fit(original, args.cell_size, Image.Resampling.LANCZOS),
            fit(mask_preview(sample.subject_mask), args.cell_size, Image.Resampling.NEAREST),
            fit(overlay_outline(original, sample.outline_type), args.cell_size, Image.Resampling.LANCZOS),
            grid_preview(sample.grid_outline_type, args.cell_size),
        )
        top = row * row_height
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * args.cell_size, top))
        status = sample.metadata.get("reviewStatus", "unknown")
        truncated = sample.metadata.get("isTruncated", False)
        draw.text((8, top + args.cell_size + 7), f"{sample.sample_id} | {status} | truncated={truncated}", fill="black")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="PNG")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

