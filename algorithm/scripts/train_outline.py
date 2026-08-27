#!/usr/bin/env python3
"""Train the first bootstrap outline model."""

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

from outline_ai.data.torch_dataset import TorchOutlineDataset, load_split_ids  # noqa: E402
from outline_ai.losses import OutlineLoss  # noqa: E402
from outline_ai.models import OutlineModel, OutlineModelConfig  # noqa: E402
from outline_ai.training import OutlineTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--tiny-smoke-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    train_ids = load_split_ids(args.split_manifest, "train")
    validation_ids = load_split_ids(args.split_manifest, "validation")
    size = (args.input_size, args.input_size)
    train_loader = DataLoader(
        TorchOutlineDataset(args.dataset, train_ids, size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        TorchOutlineDataset(args.dataset, validation_ids, size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    config = OutlineModelConfig(
        encoder_channels=(8, 16, 32, 64) if args.tiny_smoke_model else (32, 64, 128, 256),
        decoder_channels=16 if args.tiny_smoke_model else 64,
    )
    model = OutlineModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    trainer = OutlineTrainer(model, OutlineLoss(), optimizer, device)
    best_loss = float("inf")
    args.output.mkdir(parents=True, exist_ok=True)
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
