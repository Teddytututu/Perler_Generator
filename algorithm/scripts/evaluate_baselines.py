#!/usr/bin/env python3
"""Evaluate deterministic baselines on standardized outline samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.baselines import CannyBaseline, MaskBoundaryBaseline  # noqa: E402
from outline_ai.data.dataset import discover_samples, load_sample  # noqa: E402
from outline_ai.metrics import binary_metrics, contour_continuity, evaluate_closure, smoothness_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("canny", "mask_boundary"), default=("canny", "mask_boundary"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tolerance", type=int, default=1)
    return parser.parse_args()


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    numeric = ("outer_f1", "continuity", "smoothness", "isolated_pixel_ratio")
    result: dict[str, object] = {"samples": len(records)}
    for name in numeric:
        result[name] = float(np.mean([float(record[name]) for record in records])) if records else None
    result["strict_closure_rate"] = float(np.mean([bool(record["strict_closed"]) for record in records])) if records else None
    result["repairable_closure_rate"] = float(np.mean([bool(record["repairable_closed"]) for record in records])) if records else None
    return result


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    predictions = output / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    paths = discover_samples(args.dataset)
    if args.limit is not None:
        paths = paths[: args.limit]
    methods = {
        "canny": CannyBaseline(),
        "mask_boundary": MaskBoundaryBaseline(),
    }
    all_records: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    for method_name in args.methods:
        method_records: list[dict[str, object]] = []
        method_dir = predictions / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            sample = load_sample(path)
            if method_name == "canny":
                prediction = methods[method_name].predict(sample.image)
            else:
                prediction = methods[method_name].predict(sample.subject_mask)
            target_grid = sample.grid_outline_type == 1
            pixels = binary_metrics(prediction.grid_outline, target_grid, tolerance=args.tolerance)
            closure = evaluate_closure(
                prediction.grid_outline,
                allowed_gap=int(sample.metadata.get("allowedGapGrid", 1)),
            )
            smoothness = smoothness_score(prediction.grid_outline)
            record = {
                "method": method_name,
                "sample_id": sample.sample_id,
                "outer_f1": pixels.f1,
                "precision": pixels.precision,
                "recall": pixels.recall,
                "continuity": contour_continuity(prediction.grid_outline, target_grid, args.tolerance),
                "strict_closed": closure.strict_closed,
                "repairable_closed": closure.repairable_closed,
                "endpoint_count": closure.endpoint_count,
                "smoothness": smoothness.score,
                "isolated_pixel_ratio": smoothness.isolated_pixel_ratio,
            }
            method_records.append(record)
            all_records.append(record)
            Image.fromarray(prediction.grid_outline.astype(np.uint8) * 255, mode="L").save(
                method_dir / f"{sample.sample_id}.png"
            )
        summaries[method_name] = aggregate(method_records)

    with (output / "baseline_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summaries, "samples": all_records}, handle, indent=2)
        handle.write("\n")
    with (output / "baseline_report.csv").open("w", encoding="utf-8", newline="") as handle:
        if all_records:
            writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
            writer.writeheader()
            writer.writerows(all_records)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

