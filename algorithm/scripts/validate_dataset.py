#!/usr/bin/env python3
"""Validate a standardized outline dataset and write JSON/CSV reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from outline_ai.data.dataset import discover_samples  # noqa: E402
from outline_ai.data.validators import summarize_validations, validate_sample_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = discover_samples(args.dataset)
    records = [validate_sample_dir(path) for path in samples]
    summary = summarize_validations(records)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with (output / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "samples": [record.to_dict() for record in records]}, handle, indent=2)
        handle.write("\n")
    with (output / "dataset_report.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "path",
            "valid",
            "issue_codes",
            "issue_severities",
            "subject_ratio",
            "outline_pixels",
            "strict_closed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "path": record.path,
                    "valid": record.valid,
                    "issue_codes": ";".join(issue.code for issue in record.issues),
                    "issue_severities": ";".join(issue.severity for issue in record.issues),
                    "subject_ratio": record.subject_ratio,
                    "outline_pixels": record.outline_pixels,
                    "strict_closed": record.strict_closed,
                }
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
