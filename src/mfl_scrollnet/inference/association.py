"""One-to-one geometric association with novelty-aware initialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mfl_scrollnet.config import AssociationConfig
from mfl_scrollnet.utils.boxes import box_iou


@dataclass(slots=True)
class GlobalDetection:
    track_id: int
    box: torch.Tensor
    class_id: int
    score: float
    first_window: int
    last_window: int
    observations: int = 1
    fusion_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.fusion_weight <= 0:
            self.fusion_weight = max(self.score, 1e-8)

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "bbox_xyxy": self.box.detach().cpu().tolist(),
            "class_id": self.class_id,
            "score": self.score,
            "first_window": self.first_window,
            "last_window": self.last_window,
            "observations": self.observations,
        }


class GlobalAssociator:
    """Maintain active global objects and fuse duplicate window observations."""

    def __init__(self, config: AssociationConfig) -> None:
        self.config = config
        self.active: list[GlobalDetection] = []
        self.completed: list[GlobalDetection] = []
        self._next_track_id = 0

    def reset(self) -> None:
        self.active.clear()
        self.completed.clear()
        self._next_track_id = 0

    def _expire(self, window_index: int) -> None:
        retained: list[GlobalDetection] = []
        for item in self.active:
            if window_index - item.last_window > self.config.max_inactive_windows:
                self.completed.append(item)
            else:
                retained.append(item)
        self.active = retained

    def update(self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
               novelty: torch.Tensor, window_index: int) -> None:
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("Boxes must have shape [N, 4]")
        count = boxes.shape[0]
        if any(item.ndim != 1 or item.shape[0] != count for item in (scores, labels, novelty)):
            raise ValueError("Scores, labels, and novelty must have shape [N]")
        if window_index < 0:
            raise ValueError("Window index cannot be negative")
        self._expire(window_index)
        if boxes.numel() == 0:
            return
        active_boxes = (
            torch.stack([item.box for item in self.active])
            if self.active else boxes.new_empty((0, 4))
        )
        active_labels = torch.tensor([item.class_id for item in self.active], device=labels.device,
                                     dtype=labels.dtype)
        overlaps = box_iou(boxes, active_boxes)
        if active_labels.numel():
            overlaps = overlaps.masked_fill(labels[:, None] != active_labels[None, :], -1)

        candidates: list[tuple[float, int, int]] = []
        for detection_index, active_index in torch.nonzero(
                overlaps >= self.config.iou_threshold, as_tuple=False).tolist():
            candidates.append((float(overlaps[detection_index, active_index]),
                               detection_index, active_index))
        candidates.sort(reverse=True)
        matched_detections: set[int] = set()
        matched_active: set[int] = set()
        for _, detection_index, active_index in candidates:
            if detection_index in matched_detections or active_index in matched_active:
                continue
            self._fuse(self.active[active_index], boxes[detection_index],
                       float(scores[detection_index]), window_index)
            matched_detections.add(detection_index)
            matched_active.add(active_index)

        for index in range(boxes.shape[0]):
            if index in matched_detections:
                continue

            if float(novelty[index]) < self.config.novelty_threshold and self.active:
                continue
            self.active.append(GlobalDetection(
                track_id=self._next_track_id,
                box=boxes[index].detach().clone(),
                class_id=int(labels[index]),
                score=float(scores[index]),
                first_window=window_index,
                last_window=window_index,
            ))
            self._next_track_id += 1

    def _fuse(self, track: GlobalDetection, box: torch.Tensor, score: float,
              window_index: int) -> None:
        if self.config.confidence_weighted_fusion:
            denominator = max(track.fusion_weight + score, 1e-8)
            track.box = (
                track.box * track.fusion_weight + box.detach() * score
            ) / denominator
            track.fusion_weight = denominator
        elif score > track.score:
            track.box = box.detach().clone()
        track.score = max(track.score, score)
        track.last_window = window_index
        track.observations += 1

    def finish(self) -> list[GlobalDetection]:
        result = self.completed + self.active
        return sorted(result, key=lambda item: item.score, reverse=True)

    def statistics(self) -> dict[str, int]:
        tracks = self.completed + self.active
        return {
            "tracks": len(tracks),
            "active_tracks": len(self.active),
            "completed_tracks": len(self.completed),
            "observations": sum(item.observations for item in tracks),
        }
