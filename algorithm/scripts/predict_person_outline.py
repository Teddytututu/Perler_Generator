#!/usr/bin/env python3
"""Generate a clean person mask and outline from an ordinary RGB image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.data.rasterize import mask_to_outer_boundary, project_mask_to_grid_boundary  # noqa: E402
from outline_ai.models import PretrainedPersonSegmenter  # noqa: E402
from outline_ai.postprocess import clean_subject_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def prepare_image(image: Image.Image, size: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, (left, top))
    value = torch.from_numpy(np.asarray(canvas).copy().transpose(2, 0, 1)).float() / 255.0
    value = (value - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225]
    )[:, None, None]
    return value.unsqueeze(0), (left, top, resized.width, resized.height)


def restore_map(value: np.ndarray, box: tuple[int, int, int, int], original_size: tuple[int, int]) -> np.ndarray:
    left, top, width, height = box
    cropped = value[top : top + height, left : left + width]
    image = Image.fromarray(cropped.astype(np.float32), mode="F")
    return np.asarray(image.resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32)


def save_binary(path: Path, value: np.ndarray, scale: int = 1) -> None:
    image = Image.fromarray((value > 0).astype(np.uint8) * 255, mode="L")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    image.save(path)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    with Image.open(args.input) as raw:
        original = ImageOps.exif_transpose(raw).convert("RGB")
    tensor, box = prepare_image(original, args.input_size)
    model = PretrainedPersonSegmenter(pretrained=True)
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("recommended_threshold", 0.8)) if checkpoint else 0.05
    if not 0.0 < threshold < 1.0:
        raise SystemExit("threshold must be in (0, 1)")
    model.to(device).eval()
    with torch.inference_mode():
        probability_square = model(tensor.to(device))["subject_logits"].sigmoid()[0, 0].cpu().numpy()
    probability = restore_map(probability_square, box, original.size)
    mask = clean_subject_mask(probability >= threshold)
    outline = mask_to_outer_boundary(mask)
    grid = project_mask_to_grid_boundary(mask, (args.grid_size, args.grid_size))
    args.output.mkdir(parents=True, exist_ok=True)
    original.save(args.output / "image.png")
    Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L").save(
        args.output / "probability.png"
    )
    save_binary(args.output / "subject_mask.png", mask)
    save_binary(args.output / "outline.png", outline)
    save_binary(args.output / "grid_outline.png", grid, scale=8)
    overlay = np.asarray(original).copy()
    overlay[outline > 0] = np.array([255, 0, 0], dtype=np.uint8)
    Image.fromarray(overlay).save(args.output / "overlay.png")
    summary = {
        "input": args.input.as_posix(),
        "checkpoint": args.checkpoint.as_posix() if args.checkpoint else "torchvision_pretrained",
        "threshold": threshold,
        "subject_ratio": float(mask.mean()),
        "outline_pixels": int(outline.sum()),
        "grid_outline_pixels": int(grid.sum()),
    }
    with (args.output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
