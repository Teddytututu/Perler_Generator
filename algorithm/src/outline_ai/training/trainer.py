"""Small reproducible training loop with validation and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


class OutlineTrainer:
    def __init__(self, model, loss_fn, optimizer, device: torch.device) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device

    def _move_batch(self, batch: dict) -> dict:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def run_epoch(self, loader: Iterable[dict], training: bool, max_steps: int | None = None) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        steps = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                batch = self._move_batch(batch)
                prediction = self.model(batch["image"])
                losses = self.loss_fn(prediction, batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    self.optimizer.step()
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
        if steps == 0:
            raise ValueError("Cannot run an epoch with an empty data loader")
        return {name: value / steps for name, value in totals.items()}

    def save_checkpoint(self, path: Path, epoch: int, validation_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
