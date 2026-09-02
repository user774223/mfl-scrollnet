"""End-to-end MFL-ScrollNet window step."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mfl_scrollnet.config import ExperimentConfig
from mfl_scrollnet.utils.boxes import apply_deltas, clip_boxes

from .backbone import DarknetFPN
from .recurrent import ObjectContextModule
from .roi import roi_align
from .yolo import DenseDetections, YOLOHead, decode_predictions


@dataclass(slots=True)
class WindowOutput:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    novelty: torch.Tensor
    embeddings_count: int
    contextual_objectness: torch.Tensor
    class_probabilities: torch.Tensor


class MFLScrollNet(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        model = config.model
        self.backbone = DarknetFPN(config.data.input_channels, model.base_channels)
        anchors_per_scale = len(model.anchors) // 3
        self.detector = YOLOHead(self.backbone.out_channels, anchors_per_scale,
                                 len(config.data.class_names))
        self.context = ObjectContextModule(
            self.backbone.out_channels[0], model.roi_size, model.embedding_dim,
            model.hidden_dim, model.context_dim, len(config.data.class_names),
        )
        self.register_buffer("anchors", torch.tensor(model.anchors, dtype=torch.float32))

    def forward_raw(self, images: torch.Tensor) -> tuple[tuple[torch.Tensor, ...],
                                                          tuple[torch.Tensor, ...]]:
        features = self.backbone(images)
        return self.detector(features), features

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.context.initial_state(batch_size, device, dtype)

    def forward_window(self, images: torch.Tensor, previous_state: torch.Tensor,
                       proposals: list[DenseDetections] | None = None) -> tuple[
                           list[WindowOutput], torch.Tensor, tuple[torch.Tensor, ...]]:
        raw, features = self.forward_raw(images)
        image_size = images.shape[-2:]
        if proposals is None:
            proposals = decode_predictions(
                raw, self.anchors, image_size, self.config.model.score_threshold,
                self.config.model.nms_iou_threshold,
                self.config.model.max_detections_per_window,
            )
        if len(proposals) != images.shape[0]:
            raise ValueError("Proposal batch size must match the image batch size")
        boxes = [item.boxes for item in proposals]
        roi_features = roi_align(features[0], boxes, self.config.model.roi_size, image_size)
        counts = [item.boxes.shape[0] for item in proposals]
        hidden, deltas, contextual_objectness, classes, novelty = self.context(
            roi_features, counts, previous_state
        )
        outputs: list[WindowOutput] = []
        cursor = 0
        image_height, image_width = image_size
        for proposal, count in zip(proposals, counts, strict=True):
            refined = apply_deltas(proposal.boxes, deltas[cursor:cursor + count])
            if count:
                refined = clip_boxes(refined, image_width, image_height)
                class_probability, labels = classes[cursor:cursor + count].max(dim=-1)
                scores = (
                    proposal.scores
                    * contextual_objectness[cursor:cursor + count]
                    * class_probability
                )
            else:
                labels = proposal.labels
                scores = proposal.scores
            outputs.append(WindowOutput(
                boxes=refined,
                scores=scores,
                labels=labels,
                novelty=novelty[cursor:cursor + count],
                embeddings_count=count,
                contextual_objectness=contextual_objectness[cursor:cursor + count],
                class_probabilities=classes[cursor:cursor + count],
            ))
            cursor += count
        return outputs, hidden, raw

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        self.backbone.eval()
        self.detector.eval()

    def unfreeze_backbone(self) -> None:
        """Restore end-to-end training after a frozen sequential stage."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        for parameter in self.detector.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> MFLScrollNet:
        super().train(mode)
        if mode and not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
            self.detector.eval()
        return self
