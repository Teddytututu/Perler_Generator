#!/usr/bin/env python3
"""Evaluate person segmentation on a held-out split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.data.torch_dataset import PersonSegmentationDataset, load_split_ids  # noqa: E402
from outline_ai.models import PretrainedPersonSegmenter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    ids = load_split_ids(args.split_manifest, args.split)
    loader = DataLoader(
        PersonSegmentationDataset(args.dataset, ids, (args.input_size, args.input_size)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = PretrainedPersonSegmenter(pretrained=True)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    records = []
    with torch.inference_mode():
        for batch in loader:
            probability = model(batch["image"].to(device))["subject_logits"].sigmoid().cpu()
            predicted = probability >= args.threshold
            expected = batch["subject"] > 0.5
            for index, sample_id in enumerate(batch["sample_id"]):
                pred = predicted[index]
                truth = expected[index]
                intersection = int((pred & truth).sum())
                union = int((pred | truth).sum())
                predicted_count = int(pred.sum())
                expected_count = int(truth.sum())
                records.append(
                    {
                        "sample_id": sample_id,
                        "iou": intersection / union if union else 1.0,
                        "precision": intersection / predicted_count if predicted_count else float(expected_count == 0),
                        "recall": intersection / expected_count if expected_count else 1.0,
                    }
                )
    summary = {
        "samples": len(records),
        "mean_iou": sum(record["iou"] for record in records) / len(records),
        "mean_precision": sum(record["precision"] for record in records) / len(records),
        "mean_recall": sum(record["recall"] for record in records) / len(records),
        "iou_below_0_5": sum(record["iou"] < 0.5 for record in records),
        "iou_below_0_75": sum(record["iou"] < 0.75 for record in records),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "segmentation_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "samples": records}, handle, indent=2)
        handle.write("\n")
    with (args.output / "segmentation_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "iou", "precision", "recall"))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
