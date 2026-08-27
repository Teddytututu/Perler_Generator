from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_dataset_split.py"
SPEC = importlib.util.spec_from_file_location("create_dataset_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_create_split_is_deterministic_and_disjoint() -> None:
    sample_ids = [f"sample_{index:04d}" for index in range(100)]
    first = MODULE.create_split(sample_ids, 7, 0.8, 0.1)
    second = MODULE.create_split(sample_ids, 7, 0.8, 0.1)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
