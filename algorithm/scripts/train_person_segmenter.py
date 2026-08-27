#!/usr/bin/env python3
"""Fine-tune pretrained person segmentation on anime image/mask pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.data.torch_dataset import PersonSegmentationDataset, load_split_ids  # noqa: E402
from outline_ai.losses import PersonSegmentationLoss  # noqa: E402
from outline_ai.models import PretrainedPersonSegmenter  # noqa: E402
from outline_ai.training import OutlineTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    size = (args.input_size, args.input_size)
    train_loader = DataLoader(
        PersonSegmentationDataset(
            args.dataset,
            load_split_ids(args.split_manifest, "train"),
            size,
            horizontal_flip_probability=0.5,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    validation_loader = DataLoader(
        PersonSegmentationDataset(args.dataset, load_split_ids(args.split_manifest, "validation"), size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = PretrainedPersonSegmenter(pretrained=True)
    if args.freeze_backbone:
        model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    trainer = OutlineTrainer(model, PersonSegmentationLoss(), optimizer, device)
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.run_epoch(train_loader, True, args.max_train_steps)
        validation_metrics = trainer.run_epoch(validation_loader, False, args.max_validation_steps)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        trainer.save_checkpoint(args.output / "last.pt", epoch, validation_metrics["total"])
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            trainer.save_checkpoint(args.output / "best.pt", epoch, best_loss)
        with (args.output / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")
        print(json.dumps(record))


if __name__ == "__main__":
    main()
