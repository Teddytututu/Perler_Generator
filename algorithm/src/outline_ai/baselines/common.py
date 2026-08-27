"""Shared baseline result contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineOutput:
    highres_outline: np.ndarray
    grid_outline: np.ndarray
    diagnostics: dict[str, object]

