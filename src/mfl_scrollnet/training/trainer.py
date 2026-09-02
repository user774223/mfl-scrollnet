from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from mfl_scrollnet.config import ExperimentConfig
from mfl_scrollnet.data.dataset import WindowSample
from mfl_scrollnet.models import MFLScrollNet
from mfl_scrollnet.models.yolo import DenseDetections
from mfl_scrollnet.utils.tensors import ensure_finite

from .checkpoint import load_checkpoint, save_checkpoint
from .losses import DetectionLoss, SequentialLoss
from .targets import build_yolo_targets


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    stage: str
    metrics: dict[str, float]
    learning_rate: float


class Trainer:
    def __init__(self, model: MFLScrollNet, config: ExperimentConfig,
                 device: torch.device | str) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.training.restart_period,
            T_mult=config.training.restart_multiplier,
        )
        self.detection_loss = DetectionLoss(config.training)
        self.sequential_loss = SequentialLoss(config.training)
        amp_enabled = config.training.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.epoch = -1
        self.stage = "uninitialized"

    def _step(self, loss: torch.Tensor) -> None:
        ensure_finite(loss, "training loss")
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def train_local_epoch(
        self,
        loader: Iterable[tuple[torch.Tensor, list[dict[str, torch.Tensor]]]],
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        totals = {key: 0.0 for key in ("total", "box", "objectness", "classification")}
        batches = 0
        for batch_index, (images, targets) in enumerate(loader):
            images = images.to(self.device)
            with torch.autocast(self.device.type, enabled=self.scaler.is_enabled()):
                raw, _ = self.model.forward_raw(images)
                encoded = build_yolo_targets(
                    raw, targets, self.model.anchors, images.shape[-2:],
                    len(self.config.data.class_names),
                )
                breakdown = self.detection_loss(raw, encoded)
            self._step(breakdown.total)
            for key in totals:
                totals[key] += breakdown.scalars()[key]
            batches += 1
            self.scheduler.step(epoch + batch_index / max(len(loader), 1))
        return {key: value / max(batches, 1) for key, value in totals.items()}

    def train_sequential_epoch(self, loader: Iterable[list[list[WindowSample]]],
                               epoch: int) -> dict[str, float]:
        self.model.train()
        if self.config.training.freeze_backbone_in_stage2:
            self.model.freeze_backbone()
        totals = {key: 0.0 for key in
                  ("total", "box", "objectness", "classification", "novelty")}
        sequences = 0
        for batch_index, batch in enumerate(loader):
            batch_loss = None
            for sequence in batch:
                hidden = self.model.initial_state(1, self.device)
                outputs = []
                targets = []
                for sample in sequence:
                    image = sample.image.unsqueeze(0).to(self.device)
                    boxes = sample.boxes.to(self.device)
                    labels = sample.labels.to(self.device)
                    teacher = [DenseDetections(
                        boxes=boxes,
                        scores=torch.ones(boxes.shape[0], device=self.device),
                        labels=labels,
                    )]
                    window_outputs, hidden, _ = self.model.forward_window(image, hidden, teacher)
                    outputs.append(window_outputs[0])
                    targets.append({
                        "boxes": boxes,
                        "labels": labels,
                        "novelty": sample.novelty.to(self.device),
                    })
                breakdown = self.sequential_loss(outputs, targets)
                batch_loss = breakdown.total if batch_loss is None else batch_loss + breakdown.total
                for key in totals:
                    totals[key] += breakdown.scalars()[key]
                sequences += 1
            if batch_loss is not None:
                self._step(batch_loss / max(len(batch), 1))
            self.scheduler.step(epoch + batch_index / max(len(loader), 1))
        return {key: value / max(sequences, 1) for key, value in totals.items()}

    def checkpoint(self, path: str | Path, epoch: int, stage: str,
                   metrics: dict[str, float]) -> None:
        save_checkpoint(path, {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "metrics": metrics,
            "config": self.config.to_dict(),
        })

    def resume(self, path: str | Path, strict: bool = True) -> dict[str, object]:
        """Restore complete training state and return checkpoint metadata."""
        payload = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.scaler, strict=strict
        )
        self.epoch = int(payload.get("epoch", -1))
        self.stage = str(payload.get("stage", "unknown"))
        return payload

    def fit_local(self, loader: Iterable, epochs: int | None = None,
                  checkpoint_dir: str | Path | None = None,
                  callback: Callable[[EpochResult], None] | None = None) -> list[EpochResult]:
        """Run Stage 1 with optional per-epoch checkpoints and progress callbacks."""
        return self._fit(
            stage="local", loader=loader,
            epochs=self.config.training.epochs_local if epochs is None else epochs,
            train_epoch=self.train_local_epoch,
            checkpoint_dir=checkpoint_dir, callback=callback,
        )

    def fit_sequential(self, loader: Iterable, epochs: int | None = None,
                       checkpoint_dir: str | Path | None = None,
                       callback: Callable[[EpochResult], None] | None = None) -> list[EpochResult]:
        """Run Stage 2 with truncated recurrent sequences."""
        return self._fit(
            stage="sequential", loader=loader,
            epochs=self.config.training.epochs_sequential if epochs is None else epochs,
            train_epoch=self.train_sequential_epoch,
            checkpoint_dir=checkpoint_dir, callback=callback,
        )

    def _fit(self, stage: str, loader: Iterable, epochs: int,
             train_epoch: Callable[[Iterable, int], dict[str, float]],
             checkpoint_dir: str | Path | None,
             callback: Callable[[EpochResult], None] | None) -> list[EpochResult]:
        if epochs <= 0:
            raise ValueError("Epoch count must be positive")
        history: list[EpochResult] = []
        first_epoch = self.epoch + 1 if self.stage == stage else 0
        for epoch in range(first_epoch, first_epoch + epochs):
            metrics = train_epoch(loader, epoch)
            self.epoch, self.stage = epoch, stage
            result = EpochResult(epoch, stage, metrics, self.learning_rate)
            history.append(result)
            if checkpoint_dir is not None:
                directory = Path(checkpoint_dir)
                self.checkpoint(directory / "latest.pt", epoch, stage, metrics)
            if callback is not None:
                callback(result)
        return history
