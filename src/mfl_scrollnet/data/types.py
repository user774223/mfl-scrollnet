"""Data structures shared by loading, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class Annotation:
    """One global annotation in XYXY panorama coordinates."""

    box: tuple[float, float, float, float]
    class_id: int
    object_id: int

    def __post_init__(self) -> None:
        if len(self.box) != 4:
            raise ValueError("Annotation boxes must contain four XYXY coordinates")
        x1, y1, x2, y2 = self.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid annotation box: {self.box}")
        if self.class_id < 0:
            raise ValueError("Class IDs cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox_xyxy": list(self.box),
            "class_id": self.class_id,
            "object_id": self.object_id,
        }


@dataclass(frozen=True, slots=True)
class PanoramaRecord:
    image_path: Path
    panorama_id: str
    annotations: tuple[Annotation, ...]

    def __post_init__(self) -> None:
        if not self.panorama_id:
            raise ValueError("Panorama ID cannot be empty")
        object_ids = [item.object_id for item in self.annotations]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f"Duplicate object IDs in panorama {self.panorama_id!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.panorama_id,
            "image": str(self.image_path),
            "annotations": [item.to_dict() for item in self.annotations],
        }


@dataclass(slots=True)
class WindowSample:
    image: torch.Tensor
    boxes: torch.Tensor
    labels: torch.Tensor
    object_ids: torch.Tensor
    novelty: torch.Tensor
    start_x: int
    panorama_id: str
    valid_width: int

    @property
    def target_count(self) -> int:
        return int(self.boxes.shape[0])

    def to(self, device: torch.device | str, non_blocking: bool = False) -> WindowSample:
        """Move tensors in-place while retaining window metadata."""
        self.image = self.image.to(device, non_blocking=non_blocking)
        self.boxes = self.boxes.to(device, non_blocking=non_blocking)
        self.labels = self.labels.to(device, non_blocking=non_blocking)
        self.object_ids = self.object_ids.to(device, non_blocking=non_blocking)
        self.novelty = self.novelty.to(device, non_blocking=non_blocking)
        return self
