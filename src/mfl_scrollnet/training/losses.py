"""Stage-specific training objectives from the method description."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mfl_scrollnet.config import TrainingConfig
from mfl_scrollnet.models.scrollnet import WindowOutput


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    box: torch.Tensor
    objectness: torch.Tensor
    classification: torch.Tensor
    novelty: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach())
                for name in ("total", "box", "objectness", "classification", "novelty")}


class DetectionLoss(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, predictions: tuple[torch.Tensor, ...],
                targets: tuple[torch.Tensor, ...]) -> LossBreakdown:
        box = predictions[0].new_zeros(())
        objectness = predictions[0].new_zeros(())
        classification = predictions[0].new_zeros(())
        positives = 0
        for prediction, target in zip(predictions, targets, strict=True):
            positive = target[..., 4] > 0
            objectness = objectness + F.binary_cross_entropy_with_logits(
                prediction[..., 4], target[..., 4], reduction="mean"
            )
            if positive.any():
                positives += int(positive.sum())
                box = box + F.binary_cross_entropy_with_logits(
                    prediction[..., :2][positive], target[..., :2][positive], reduction="sum"
                )
                box = box + F.smooth_l1_loss(
                    prediction[..., 2:4][positive], target[..., 2:4][positive], reduction="sum"
                )
                class_ids = target[..., 5:][positive].argmax(dim=-1)
                classification = classification + F.cross_entropy(
                    prediction[..., 5:][positive], class_ids, reduction="sum"
                )
        normalizer = max(positives, 1)
        box = box / normalizer
        classification = classification / normalizer
        total = (self.config.box_weight * box + self.config.objectness_weight * objectness
                 + self.config.class_weight * classification)
        zero = total.new_zeros(())
        return LossBreakdown(total, box, objectness, classification, zero)


class SequentialLoss(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, outputs: list[WindowOutput],
                targets: list[dict[str, torch.Tensor]]) -> LossBreakdown:
        if not outputs:
            raise ValueError("Sequential loss requires at least one window output")
        device = outputs[0].boxes.device
        graph_zero = sum((output.novelty.sum() * 0 for output in outputs),
                         torch.zeros((), device=device))
        box = graph_zero
        objectness = torch.zeros((), device=device)
        classification = torch.zeros((), device=device)
        novelty = torch.zeros((), device=device)
        count = 0
        for output, target in zip(outputs, targets, strict=True):
            target_boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            novelty_target = target["novelty"].to(device)
            if target_boxes.numel() == 0:
                continue
            if output.boxes.shape[0] != target_boxes.shape[0]:
                raise ValueError("Stage 2 expects one teacher proposal per target object")
            count += target_boxes.shape[0]
            box = box + F.smooth_l1_loss(output.boxes, target_boxes, reduction="sum")
            objectness = objectness + F.binary_cross_entropy(
                output.contextual_objectness, torch.ones_like(output.contextual_objectness),
                reduction="sum"
            )
            classification = classification + F.nll_loss(
                output.class_probabilities.clamp_min(1e-7).log(), labels, reduction="sum"
            )
            novelty = novelty + F.binary_cross_entropy(
                output.novelty, novelty_target, reduction="sum"
            )
        normalizer = max(count, 1)
        box, objectness = box / normalizer, objectness / normalizer
        classification, novelty = classification / normalizer, novelty / normalizer
        total = (self.config.box_weight * box + self.config.objectness_weight * objectness
                 + self.config.class_weight * classification
                 + self.config.novelty_weight * novelty)
        return LossBreakdown(total, box, objectness, classification, novelty)
