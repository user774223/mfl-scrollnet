from __future__ import annotations

import torch

from mfl_scrollnet.utils.boxes import xyxy_to_xywh


def _size_iou(sizes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    intersection = torch.minimum(sizes[:, None, 0], anchors[None, :, 0]) * torch.minimum(
        sizes[:, None, 1], anchors[None, :, 1]
    )
    union = sizes.prod(-1)[:, None] + anchors.prod(-1)[None, :] - intersection
    return intersection / union.clamp_min(1e-7)


def build_yolo_targets(raw_scales: tuple[torch.Tensor, ...], targets: list[dict[str, torch.Tensor]],
                       anchors: torch.Tensor, image_size: tuple[int, int],
                       num_classes: int) -> tuple[torch.Tensor, ...]:
    """Assign every visible object to its highest-IoU anchor exactly once."""
    result = [raw.new_zeros(raw.shape) for raw in raw_scales]
    anchors_per_scale = anchors.shape[0] // len(raw_scales)
    image_height, image_width = image_size
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].to(anchors.device)
        labels = target["labels"].to(anchors.device)
        if boxes.numel() == 0:
            continue
        xywh = xyxy_to_xywh(boxes)
        best_anchors = _size_iou(xywh[:, 2:], anchors).argmax(dim=1)
        for object_index, anchor_index_tensor in enumerate(best_anchors):
            anchor_index = int(anchor_index_tensor)
            scale_index = anchor_index // anchors_per_scale
            local_anchor = anchor_index % anchors_per_scale
            raw = raw_scales[scale_index]
            grid_height, grid_width = raw.shape[2:4]
            grid_x = xywh[object_index, 0] / image_width * grid_width
            grid_y = xywh[object_index, 1] / image_height * grid_height
            cell_x = min(max(int(grid_x), 0), grid_width - 1)
            cell_y = min(max(int(grid_y), 0), grid_height - 1)
            encoded = result[scale_index][batch_index, local_anchor, cell_y, cell_x]
            encoded[0] = grid_x - cell_x
            encoded[1] = grid_y - cell_y
            encoded[2:4] = (xywh[object_index, 2:] / anchors[anchor_index]).clamp_min(1e-7).log()
            encoded[4] = 1.0
            encoded[5 + labels[object_index]] = 1.0
    return tuple(result)

