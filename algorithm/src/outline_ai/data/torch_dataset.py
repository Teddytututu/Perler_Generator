"""PyTorch dataset adapter for standardized outline samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from outline_ai.data.dataset import load_sample
from outline_ai.data.rasterize import compute_direction_labels
from outline_ai.postprocess.endpoint import endpoint_mask


def load_split_ids(manifest_path: Path, split: str) -> list[str]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return list(payload["splits"][split])
    except KeyError as exc:
        raise KeyError(f"Unknown split {split!r} in {manifest_path}") from exc


def _resize(value: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value
    kwargs = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(value.unsqueeze(0), **kwargs).squeeze(0)


class TorchOutlineDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        dataset_root: Path,
        sample_ids: list[str],
        input_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sample_ids = list(sample_ids)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = load_sample(self.dataset_root / self.sample_ids[index])
        image = torch.from_numpy(sample.image.transpose(2, 0, 1)).float() / 255.0
        image = _resize(image, self.input_size, "bilinear")
        image = (image - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]

        def binary(value: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy((value > 0).astype(np.float32)).unsqueeze(0)
            return _resize(tensor, self.input_size, "nearest")

        def categorical(value: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(value.astype(np.float32)).unsqueeze(0)
            return _resize(tensor, self.input_size, "nearest").squeeze(0).long()

        grid_binary_np = (sample.grid_outline_binary > 0).astype(np.uint8)
        return {
            "sample_id": sample.sample_id,
            "image": image,
            "subject": binary(sample.subject_mask),
            "outline_binary": binary(sample.outline_binary),
            "outline_type": categorical(sample.outline_type),
            "direction": categorical(sample.direction),
            "endpoint": binary(sample.endpoint),
            "grid_outline": torch.from_numpy(grid_binary_np.astype(np.float32)).unsqueeze(0),
            "grid_type": torch.from_numpy(sample.grid_outline_type.astype(np.int64)),
            "grid_direction": torch.from_numpy(compute_direction_labels(grid_binary_np).astype(np.int64)),
            "grid_endpoint": torch.from_numpy(endpoint_mask(grid_binary_np).astype(np.float32)).unsqueeze(0),
        }


class PersonSegmentationDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        dataset_root: Path,
        sample_ids: list[str],
        input_size: tuple[int, int] = (320, 320),
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        self.base = TorchOutlineDataset(dataset_root, sample_ids, input_size)
        self.horizontal_flip_probability = horizontal_flip_probability

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        full = self.base[index]
        image = full["image"]
        subject = full["subject"]
        if self.horizontal_flip_probability > 0.0 and torch.rand(()) < self.horizontal_flip_probability:
            image = torch.flip(image, dims=(-1,))
            subject = torch.flip(subject, dims=(-1,))
        return {"sample_id": full["sample_id"], "image": image, "subject": subject}
