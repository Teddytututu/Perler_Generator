"""Validation rules for standardized outline datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from outline_ai.metrics.closure import evaluate_closure

from .dataset import REQUIRED_SAMPLE_FILES, OutlineSample, load_sample


ALLOWED_OUTLINE_TYPES = {0, 1, 2, 3, 255}
REQUIRED_METADATA_FIELDS = {
    "sampleId",
    "domain",
    "sourceWidth",
    "sourceHeight",
    "subjectCount",
    "primarySubjectId",
    "isTruncated",
    "isOccluded",
    "requiresOuterClosure",
    "allowedGapGrid",
    "gridWidth",
    "gridHeight",
    "annotationVersion",
    "reviewStatus",
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class SampleValidation:
    sample_id: str
    path: str
    valid: bool
    issues: tuple[ValidationIssue, ...]
    subject_ratio: float | None
    outline_pixels: int | None
    strict_closed: bool | None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["issues"] = [asdict(issue) for issue in self.issues]
        return value


def _issue(code: str, message: str, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(code=code, message=message, severity=severity)


def validate_loaded_sample(sample: OutlineSample) -> SampleValidation:
    issues: list[ValidationIssue] = []
    metadata = sample.metadata
    sample_id = str(metadata.get("sampleId", sample.root.name))

    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing_metadata:
        issues.append(_issue("metadata_missing", ", ".join(missing_metadata)))
    if sample_id != sample.root.name:
        issues.append(_issue("sample_id_mismatch", f"metadata={sample_id}, directory={sample.root.name}"))

    highres_shape = sample.outline_binary.shape
    if sample.image.shape[:2] != highres_shape:
        issues.append(_issue("image_size_mismatch", f"image={sample.image.shape[:2]}, label={highres_shape}"))
    for name, value in (
        ("subject_mask", sample.subject_mask),
        ("outline_type", sample.outline_type),
        ("direction", sample.direction),
        ("endpoint", sample.endpoint),
    ):
        if value.shape != highres_shape:
            issues.append(_issue("highres_size_mismatch", f"{name}={value.shape}, expected={highres_shape}"))

    expected_grid = (int(metadata.get("gridHeight", 48)), int(metadata.get("gridWidth", 48)))
    for name, value in (
        ("grid_outline_type", sample.grid_outline_type),
        ("grid_outline_binary", sample.grid_outline_binary),
    ):
        if value.shape != expected_grid:
            issues.append(_issue("grid_size_mismatch", f"{name}={value.shape}, expected={expected_grid}"))

    values = set(int(item) for item in np.unique(sample.outline_type))
    invalid_types = sorted(values - ALLOWED_OUTLINE_TYPES)
    if invalid_types:
        issues.append(_issue("invalid_outline_type", str(invalid_types)))

    binary_outline = sample.outline_binary > 0
    typed_outline = np.isin(sample.outline_type, (1, 2, 3))
    if not np.array_equal(binary_outline, typed_outline):
        issues.append(_issue("binary_type_disagreement", "outline_binary and outline_type disagree"))
    if not binary_outline.any():
        issues.append(_issue("empty_outline", "outline_binary contains no contour pixels"))

    subject = sample.subject_mask > 0
    key_feature = sample.outline_type == 3
    if np.any(key_feature & ~subject):
        issues.append(_issue("key_feature_outside_subject", "key-feature pixels fall outside subject mask"))

    closure = evaluate_closure(sample.grid_outline_type == 1, allowed_gap=int(metadata.get("allowedGapGrid", 1)))
    if metadata.get("requiresOuterClosure") and not metadata.get("isTruncated") and not closure.strict_closed:
        severity = "error" if metadata.get("reviewStatus") == "approved" else "warning"
        issues.append(
            _issue(
                "outer_not_strictly_closed",
                "required grid outer contour is not strictly closed",
                severity=severity,
            )
        )

    return SampleValidation(
        sample_id=sample_id,
        path=sample.root.as_posix(),
        valid=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        subject_ratio=float(subject.mean()),
        outline_pixels=int(binary_outline.sum()),
        strict_closed=closure.strict_closed,
    )


def validate_sample_dir(sample_root: Path) -> SampleValidation:
    root = Path(sample_root)
    missing = [name for name in REQUIRED_SAMPLE_FILES if not (root / name).is_file()]
    if missing:
        return SampleValidation(
            sample_id=root.name,
            path=root.as_posix(),
            valid=False,
            issues=(_issue("missing_files", ", ".join(missing)),),
            subject_ratio=None,
            outline_pixels=None,
            strict_closed=None,
        )
    try:
        sample = load_sample(root)
    except Exception as exc:
        return SampleValidation(
            sample_id=root.name,
            path=root.as_posix(),
            valid=False,
            issues=(_issue("load_error", f"{type(exc).__name__}: {exc}"),),
            subject_ratio=None,
            outline_pixels=None,
            strict_closed=None,
        )
    return validate_loaded_sample(sample)


def summarize_validations(records: Iterable[SampleValidation]) -> dict[str, object]:
    values = list(records)
    issue_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for record in values:
        for issue in record.issues:
            issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
    closure_values = [record.strict_closed for record in values if record.strict_closed is not None]
    return {
        "total": len(values),
        "valid": sum(record.valid for record in values),
        "invalid": sum(not record.valid for record in values),
        "strict_closure_rate": (
            sum(bool(value) for value in closure_values) / len(closure_values) if closure_values else None
        ),
        "issue_counts": issue_counts,
        "severity_counts": severity_counts,
    }
