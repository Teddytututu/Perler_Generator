#!/usr/bin/env python3
"""Batch-export predicted subject masks from RGB images without reading label masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.models import PretrainedPersonSegmenter  # noqa: E402
from outline_ai.postprocess import clean_subject_mask  # noqa: E402


class RGBImageDataset(Dataset[dict[str, object]]):
    """Inference dataset that deliberately knows only about image.png."""

    def __init__(self, dataset_root: Path, input_size: int) -> None:
        self.sample_dirs = sorted(
            path
            for path in Path(dataset_root).iterdir()
            if path.is_dir() and path.name.startswith("sample_") and (path / "image.png").is_file()
        )
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_dir = self.sample_dirs[index]
        with Image.open(sample_dir / "image.png") as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            original_size = image.size
            value = torch.from_numpy(np.asarray(image).copy().transpose(2, 0, 1)).float() / 255.0
        value = F.interpolate(
            value.unsqueeze(0), size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        value = (value - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        return {
            "sample_id": sample_dir.name,
            "image": value,
            "width": original_size[0],
            "height": original_size[1],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but the installed PyTorch build has no CUDA support")
    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("recommended_threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise SystemExit("threshold must be in (0, 1)")

    dataset = RGBImageDataset(args.dataset, args.input_size)
    if not dataset:
        raise SystemExit(f"No sample image.png files found in {args.dataset}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = PretrainedPersonSegmenter(pretrained=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    exported = 0
    with torch.inference_mode():
        for batch in loader:
            probabilities = model(batch["image"].to(device))["subject_logits"].sigmoid().cpu()
            for index, sample_id in enumerate(batch["sample_id"]):
                width = int(batch["width"][index])
                height = int(batch["height"][index])
                probability = F.interpolate(
                    probabilities[index : index + 1],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                mask = clean_subject_mask(probability >= threshold)
                sample_output = args.output / sample_id
                sample_output.mkdir(parents=True, exist_ok=True)
                mask_path = sample_output / "subject_mask.png"
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
                record = {
                    "sample_id": sample_id,
                    "mask": mask_path.relative_to(args.output).as_posix(),
                    "width": width,
                    "height": height,
                    "subject_ratio": float(mask.mean()),
                    "empty": not bool(mask.any()),
                }
                records.append(record)
                exported += 1
            if exported % 100 < len(batch["sample_id"]):
                print(json.dumps({"exported": exported, "total": len(dataset)}), flush=True)

    summary = {
        "total": len(records),
        "exported": exported,
        "empty_masks": sum(bool(record["empty"]) for record in records),
        "mean_subject_ratio": sum(float(record["subject_ratio"]) for record in records) / len(records),
        "threshold": threshold,
        "checkpoint": args.checkpoint.as_posix(),
        "input_contract": "RGB image.png only; legacy masks are never opened",
    }
    with (args.output / "export_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "samples": records}, handle, indent=2)
        handle.write("\n")
    with (args.output / "export_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
