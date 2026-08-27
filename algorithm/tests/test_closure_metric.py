from __future__ import annotations

import numpy as np

from outline_ai.metrics.closure import evaluate_closure


def rectangle() -> np.ndarray:
    value = np.zeros((16, 16), dtype=np.uint8)
    value[3, 3:13] = 1
    value[12, 3:13] = 1
    value[3:13, 3] = 1
    value[3:13, 12] = 1
    return value


def test_closed_rectangle_is_strictly_closed() -> None:
    result = evaluate_closure(rectangle())
    assert result.strict_closed
    assert result.repairable_closed
    assert result.enclosed_area > 0


def test_one_cell_gap_is_repairable_but_not_strict() -> None:
    value = rectangle()
    value[3, 8] = 0
    result = evaluate_closure(value, allowed_gap=1)
    assert not result.strict_closed
    assert result.repairable_closed
    assert result.smallest_gap == 1


def test_two_cell_gap_exceeds_one_cell_allowance() -> None:
    value = rectangle()
    value[3, 8:10] = 0
    result = evaluate_closure(value, allowed_gap=1)
    assert not result.strict_closed
    assert not result.repairable_closed


def test_open_polyline_is_not_closed() -> None:
    value = np.zeros((16, 16), dtype=np.uint8)
    value[4, 4:12] = 1
    value[4:12, 11] = 1
    result = evaluate_closure(value)
    assert not result.strict_closed

