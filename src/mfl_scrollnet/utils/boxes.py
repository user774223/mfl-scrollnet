"""Vectorized box transformations, IoU, and class-wise NMS."""

from __future__ import annotations

import torch


def _validate_boxes(boxes: torch.Tensor, name: str = "boxes") -> None:
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [..., 4], got {tuple(boxes.shape)}")
    if not boxes.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Return non-negative areas for XYXY boxes."""
    _validate_boxes(boxes)
    return (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(dim=-1)


def valid_box_mask(boxes: torch.Tensor, minimum_size: float = 0.0) -> torch.Tensor:
    """Select finite boxes whose width and height exceed ``minimum_size``."""
    _validate_boxes(boxes)
    sizes = boxes[..., 2:] - boxes[..., :2]
    return torch.isfinite(boxes).all(dim=-1) & (sizes > minimum_size).all(dim=-1)


def clip_boxes(boxes: torch.Tensor, width: int | float, height: int | float) -> torch.Tensor:
    """Clip XYXY coordinates to an image without mutating the input."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    _validate_boxes(boxes)
    result = boxes.clone()
    result[..., (0, 2)] = result[..., (0, 2)].clamp(0, width)
    result[..., (1, 3)] = result[..., (1, 3)].clamp(0, height)
    return result


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    _validate_boxes(boxes)
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    _validate_boxes(boxes)
    low, high = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((low + high) / 2, high - low), dim=-1)


def box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    _validate_boxes(first, "first")
    _validate_boxes(second, "second")
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("IoU inputs must have shape [N, 4] and [M, 4]")
    if first.numel() == 0 or second.numel() == 0:
        return first.new_zeros((first.shape[0], second.shape[0]))
    intersection_low = torch.maximum(first[:, None, :2], second[None, :, :2])
    intersection_high = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (intersection_high - intersection_low).clamp_min(0).prod(dim=-1)
    first_area = box_area(first)
    second_area = box_area(second)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp_min(1e-7)


def generalized_box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    _validate_boxes(first, "first")
    _validate_boxes(second, "second")
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("GIoU inputs must have shape [N, 4] and [M, 4]")
    if first.numel() == 0 or second.numel() == 0:
        return first.new_zeros((first.shape[0], second.shape[0]))
    iou = box_iou(first, second)
    enclosure_low = torch.minimum(first[:, None, :2], second[None, :, :2])
    enclosure_high = torch.maximum(first[:, None, 2:], second[None, :, 2:])
    enclosure = (enclosure_high - enclosure_low).clamp_min(0).prod(dim=-1)
    first_area = box_area(first)
    second_area = box_area(second)
    intersection_low = torch.maximum(first[:, None, :2], second[None, :, :2])
    intersection_high = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (intersection_high - intersection_low).clamp_min(0).prod(dim=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return iou - (enclosure - union) / enclosure.clamp_min(1e-7)


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    _validate_boxes(boxes)
    if boxes.ndim != 2:
        raise ValueError("NMS boxes must have shape [N, 4]")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("Scores must have shape [N] and match boxes")
    if not 0 <= threshold <= 1:
        raise ValueError("NMS threshold must be in [0, 1]")
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        overlap = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][overlap <= threshold]
    return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                threshold: float) -> torch.Tensor:
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError("Labels must have shape [N] and match boxes")
    kept = [
        indices[nms(boxes[indices], scores[indices], threshold)]
        for label in labels.unique()
        if (indices := torch.where(labels == label)[0]).numel() > 0
    ]
    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    result = torch.cat(kept)
    return result[scores[result].argsort(descending=True)]


def apply_deltas(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply standard center/scale offsets to XYXY proposal boxes."""
    _validate_boxes(boxes)
    _validate_boxes(deltas, "deltas")
    if boxes.shape != deltas.shape:
        raise ValueError("Boxes and deltas must have identical shapes")
    original = xyxy_to_xywh(boxes)
    center = original[..., :2] + deltas[..., :2] * original[..., 2:]
    size = original[..., 2:] * deltas[..., 2:].clamp(-4, 4).exp()
    return xywh_to_xyxy(torch.cat((center, size), dim=-1))
