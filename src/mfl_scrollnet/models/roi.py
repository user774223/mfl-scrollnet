"""Dependency-free differentiable RoIAlign based on grid sampling."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def roi_align(features: torch.Tensor, boxes: list[torch.Tensor], output_size: int,
              image_size: tuple[int, int]) -> torch.Tensor:
    """Sample evenly spaced bin centers for each XYXY box."""
    channels = features.shape[1]
    feature_height, feature_width = features.shape[-2:]
    image_height, image_width = image_size
    pooled: list[torch.Tensor] = []
    for batch_index, image_boxes in enumerate(boxes):
        for box in image_boxes:
            x1, y1, x2, y2 = box.unbind()
            xs = torch.linspace(0.5 / output_size, 1 - 0.5 / output_size, output_size,
                                device=features.device, dtype=features.dtype)
            ys = torch.linspace(0.5 / output_size, 1 - 0.5 / output_size, output_size,
                                device=features.device, dtype=features.dtype)
            sample_x = (x1 + xs * (x2 - x1)) * feature_width / image_width
            sample_y = (y1 + ys * (y2 - y1)) * feature_height / image_height
            grid_y, grid_x = torch.meshgrid(sample_y, sample_x, indexing="ij")
            normalized_x = grid_x / max(feature_width - 1, 1) * 2 - 1
            normalized_y = grid_y / max(feature_height - 1, 1) * 2 - 1
            grid = torch.stack((normalized_x, normalized_y), dim=-1).unsqueeze(0)
            pooled.append(F.grid_sample(features[batch_index:batch_index + 1], grid,
                                        mode="bilinear", align_corners=True)[0])
    if not pooled:
        return features.new_empty((0, channels, output_size, output_size))
    return torch.stack(pooled)

