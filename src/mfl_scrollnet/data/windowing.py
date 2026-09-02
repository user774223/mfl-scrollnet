"""Deterministic panorama decomposition and coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SlidingWindowPlan:
    panorama_width: int
    window_width: int
    stride: int
    initial_offset: int = 0

    def __post_init__(self) -> None:
        if self.panorama_width <= 0 or self.window_width <= 0 or self.stride <= 0:
            raise ValueError("All dimensions must be positive")
        if self.stride > self.window_width:
            raise ValueError("Stride cannot exceed window width")
        if not 0 <= self.initial_offset < self.stride:
            raise ValueError("Initial offset must be within one stride")

    @property
    def overlap(self) -> int:
        return self.window_width - self.stride

    @property
    def starts(self) -> tuple[int, ...]:
        first = min(self.initial_offset, max(0, self.panorama_width - 1))
        usable = max(0, self.panorama_width - first - self.window_width)
        count = ceil(usable / self.stride) + 1
        starts = [first + i * self.stride for i in range(count)]
        final_start = max(0, self.panorama_width - self.window_width)
        if starts[-1] < final_start:
            starts.append(final_start)
        return tuple(dict.fromkeys(starts))

    def interval(self, index: int) -> tuple[int, int]:
        start = self.starts[index]
        return start, min(start + self.window_width, self.panorama_width)


def global_to_local_xyxy(boxes: torch.Tensor, start_x: int) -> torch.Tensor:
    result = boxes.clone()
    result[..., (0, 2)] -= float(start_x)
    return result


def local_to_global_xyxy(boxes: torch.Tensor, start_x: int) -> torch.Tensor:
    result = boxes.clone()
    result[..., (0, 2)] += float(start_x)
    return result


def clip_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    result = boxes.clone()
    result[..., (0, 2)] = result[..., (0, 2)].clamp(0, width)
    result[..., (1, 3)] = result[..., (1, 3)].clamp(0, height)
    return result


def visible_mask(boxes: np.ndarray, start: int, width: int) -> np.ndarray:
    """Return the article's non-empty horizontal-intersection indicator."""
    end = start + width
    return (boxes[:, 2] > start) & (boxes[:, 0] < end)


def novelty_targets(boxes: np.ndarray, current_start: int, previous_start: int | None,
                    width: int) -> np.ndarray:
    current = visible_mask(boxes, current_start, width)
    previous = np.zeros_like(current) if previous_start is None else visible_mask(
        boxes, previous_start, width
    )
    return (current & ~previous).astype(np.float32)

