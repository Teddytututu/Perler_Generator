#!/usr/bin/env python3
"""Train an RGB-image-to-subject-mask model using legacy masks as labels only."""

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
    parser = argparse.ArgumentParser(
        description=(
            "Train a model whose only inference input is an RGB image. Existing subject masks "
            "are used exclusively as supervised targets during training and evaluation."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip training and calibrate an existing locally generated best.pt checkpoint",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but the installed PyTorch build has no CUDA support")
    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def validate_split_coverage(dataset: Path, manifest: Path) -> dict[str, list[str]]:
    splits = {name: load_split_ids(manifest, name) for name in ("train", "validation", "test")}
    split_sets = {name: set(values) for name, values in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise SystemExit(f"Split leakage between {left} and {right}: {sorted(overlap)[:5]}")
    dataset_ids = {path.name for path in dataset.iterdir() if path.is_dir() and path.name.startswith("sample_")}
    manifest_ids = set().union(*split_sets.values())
    if dataset_ids != manifest_ids:
        missing = sorted(dataset_ids - manifest_ids)
        unknown = sorted(manifest_ids - dataset_ids)
        raise SystemExit(f"Split manifest does not cover the dataset: missing={missing[:5]}, unknown={unknown[:5]}")
    return splits


def make_loader(
    dataset: Path,
    sample_ids: list[str],
    size: tuple[int, int],
    batch_size: int,
    workers: int,
    training: bool,
) -> DataLoader:
    return DataLoader(
        PersonSegmentationDataset(
            dataset,
            sample_ids,
            size,
            horizontal_flip_probability=0.5 if training else 0.0,
        ),
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        drop_last=training,
    )


def calibrate_threshold(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    thresholds = [round(value, 2) for value in np.arange(0.30, 0.91, 0.05)]
    iou_totals = {threshold: 0.0 for threshold in thresholds}
    samples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            probability = model(batch["image"].to(device))["subject_logits"].sigmoid().cpu()
            expected = batch["subject"] > 0.5
            for threshold in thresholds:
                predicted = probability >= threshold
                intersection = (predicted & expected).flatten(1).sum(dim=1).float()
                union = (predicted | expected).flatten(1).sum(dim=1).float()
                iou_totals[threshold] += float(torch.where(union > 0, intersection / union, 1.0).sum())
            samples += probability.shape[0]
    mean_ious = {str(threshold): total / samples for threshold, total in iou_totals.items()}
    best = max(thresholds, key=lambda threshold: iou_totals[threshold])
    return {
        "threshold": float(best),
        "mean_iou": float(mean_ious[str(best)]),
        "samples": int(samples),
        "scores": {key: float(value) for key, value in mean_ious.items()},
    }


def main() -> None:
    args = parse_args()
    if args.warmup_epochs < 0 or args.fine_tune_epochs < 0:
        raise SystemExit("Epoch counts cannot be negative")
    if not args.finalize_only and args.warmup_epochs + args.fine_tune_epochs == 0:
        raise SystemExit("At least one training epoch is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    splits = validate_split_coverage(args.dataset, args.split_manifest)
    size = (args.input_size, args.input_size)
    train_loader = make_loader(args.dataset, splits["train"], size, args.batch_size, args.workers, True)
    validation_loader = make_loader(
        args.dataset, splits["validation"], size, args.batch_size, args.workers, False
    )

    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "input_contract": "RGB image tensor only; no mask is passed to model.forward",
        "legacy_mask_role": "training/evaluation target only",
        "dataset": args.dataset.as_posix(),
        "split_manifest": args.split_manifest.as_posix(),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "total_samples": sum(len(values) for values in splits.values()),
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "warmup_learning_rate": args.warmup_learning_rate,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "seed": args.seed,
        "device": str(device),
        "torch_version": str(torch.__version__),
    }
    with (args.output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")

    model = PretrainedPersonSegmenter(pretrained=True)
    loss_fn = PersonSegmentationLoss()
    history: list[dict[str, object]] = []
    best_loss = float("inf")
    epoch_number = 0

    phases = () if args.finalize_only else (
        ("head_warmup", args.warmup_epochs, args.warmup_learning_rate, True),
        ("full_fine_tune", args.fine_tune_epochs, args.fine_tune_learning_rate, False),
    )
    for phase_name, phase_epochs, learning_rate, freeze_backbone in phases:
        if phase_epochs == 0:
            continue
        if freeze_backbone:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)
        trainer = OutlineTrainer(model, loss_fn, optimizer, device)
        for _ in range(phase_epochs):
            epoch_number += 1
            train_metrics = trainer.run_epoch(train_loader, True)
            validation_metrics = trainer.run_epoch(validation_loader, False)
            record = {
                "epoch": epoch_number,
                "phase": phase_name,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(record)
            checkpoint = {
                "epoch": epoch_number,
                "phase": phase_name,
                "validation_loss": validation_metrics["total"],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "run_config": run_config,
            }
            torch.save(checkpoint, args.output / "last.pt")
            if validation_metrics["total"] < best_loss:
                best_loss = validation_metrics["total"]
                torch.save(checkpoint, args.output / "best.pt")
            with (args.output / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
                handle.write("\n")
            print(json.dumps(record), flush=True)
            scheduler.step()

    checkpoint_path = args.output / "best.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    # The checkpoint is generated by this local training run. weights_only=False
    # also permits repairing checkpoints written before torch_version was cast to str.
    best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = best_checkpoint.get("run_config", {})
    if "torch_version" in checkpoint_config:
        checkpoint_config["torch_version"] = str(checkpoint_config["torch_version"])
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)
    calibration = calibrate_threshold(model, validation_loader, device)
    best_checkpoint["recommended_threshold"] = calibration["threshold"]
    torch.save(best_checkpoint, checkpoint_path)
    report = {
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_loss": best_checkpoint["validation_loss"],
        "threshold_calibration": calibration,
        "input_contract": run_config["input_contract"],
        "legacy_mask_role": run_config["legacy_mask_role"],
    }
    with (args.output / "training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
