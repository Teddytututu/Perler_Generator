#!/usr/bin/env python3
"""Create a deterministic train/validation/test manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.data.dataset import discover_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def create_split(sample_ids: list[str], seed: int, train_ratio: float, validation_ratio: float) -> dict[str, list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train-ratio must be in (0, 1)")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation-ratio must be in (0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train-ratio + validation-ratio must be less than 1")
    shuffled = sorted(sample_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = round(len(shuffled) * validation_ratio)
    test_count = round(len(shuffled) * (1.0 - train_ratio - validation_ratio))
    train_count = len(shuffled) - validation_count - test_count
    return {
        "train": sorted(shuffled[:train_count]),
        "validation": sorted(shuffled[train_count : train_count + validation_count]),
        "test": sorted(shuffled[train_count + validation_count :]),
    }


def main() -> None:
    args = parse_args()
    sample_ids = [path.name for path in discover_samples(args.dataset)]
    splits = create_split(sample_ids, args.seed, args.train_ratio, args.validation_ratio)
    payload = {
        "version": 1,
        "dataset": args.dataset.as_posix(),
        "seed": args.seed,
        "grouping": "sample_id_random; source identity unavailable",
        "splits": splits,
        "counts": {name: len(values) for name, values in splits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
