from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mfl_scrollnet.utils.boxes import batched_nms, xywh_to_xyxy

from .blocks import ConvNormAct


@dataclass(slots=True)
class DenseDetections:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


class YOLOHead(nn.Module):
    def __init__(self, channels: tuple[int, int, int], anchors_per_scale: int,
                 num_classes: int) -> None:
        super().__init__()
        outputs = anchors_per_scale * (5 + num_classes)
        self.towers = nn.ModuleList([
            nn.Sequential(ConvNormAct(channel, channel, 3), nn.Conv2d(channel, outputs, 1))
            for channel in channels
        ])
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        predictions = []
        for tower, feature in zip(self.towers, features, strict=True):
            raw = tower(feature)
            batch, _, height, width = raw.shape
            predictions.append(raw.view(batch, self.anchors_per_scale,
                                        5 + self.num_classes, height, width)
                               .permute(0, 1, 3, 4, 2).contiguous())
        return tuple(predictions)


def decode_scale(raw: torch.Tensor, anchors: torch.Tensor, image_size: tuple[int, int]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, anchor_count, height, width, _ = raw.shape
    image_height, image_width = image_size
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=raw.device, dtype=raw.dtype),
        torch.arange(width, device=raw.device, dtype=raw.dtype), indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, height, width, 2)
    stride = raw.new_tensor((image_width / width, image_height / height))
    centers = (raw[..., :2].sigmoid() + grid) * stride
    sizes = raw[..., 2:4].clamp(-8, 8).exp() * anchors.view(1, anchor_count, 1, 1, 2)
    boxes = xywh_to_xyxy(torch.cat((centers, sizes), dim=-1)).view(batch, -1, 4)
    objectness = raw[..., 4].sigmoid().view(batch, -1)
    classes = raw[..., 5:].softmax(dim=-1).view(batch, -1, raw.shape[-1] - 5)
    return boxes, objectness, classes


def decode_predictions(raw_scales: tuple[torch.Tensor, ...], anchors: torch.Tensor,
                       image_size: tuple[int, int], score_threshold: float,
                       nms_threshold: float, max_detections: int) -> list[DenseDetections]:
    anchors_per_scale = anchors.shape[0] // len(raw_scales)
    decoded = [decode_scale(raw, anchors[index * anchors_per_scale:(index + 1) * anchors_per_scale],
                            image_size) for index, raw in enumerate(raw_scales)]
    boxes = torch.cat([item[0] for item in decoded], dim=1)
    objectness = torch.cat([item[1] for item in decoded], dim=1)
    classes = torch.cat([item[2] for item in decoded], dim=1)
    class_probability, labels = classes.max(dim=-1)
    scores = objectness * class_probability
    outputs: list[DenseDetections] = []
    for batch_index in range(boxes.shape[0]):
        keep = scores[batch_index] >= score_threshold
        selected_boxes = boxes[batch_index][keep]
        selected_scores = scores[batch_index][keep]
        selected_labels = labels[batch_index][keep]
        chosen = batched_nms(selected_boxes, selected_scores, selected_labels, nms_threshold)
        chosen = chosen[:max_detections]
        outputs.append(DenseDetections(selected_boxes[chosen], selected_scores[chosen],
                                       selected_labels[chosen]))
    return outputs

