"""Datasets for independent-window pre-training and ordered sequence training."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .augment import SequenceAugmenter
from .io import load_image
from .types import PanoramaRecord, WindowSample
from .windowing import SlidingWindowPlan, clip_xyxy, global_to_local_xyxy, novelty_targets


class _WindowFactory:
    def __init__(self, records: Sequence[PanoramaRecord], crop_height: int, window_width: int,
                 stride: int, input_channels: int) -> None:
        if not records:
            raise ValueError("At least one panorama record is required")
        self.records = records
        self.crop_height = crop_height
        self.window_width = window_width
        self.stride = stride
        self.input_channels = input_channels
        self._image_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def _load(self, record_index: int) -> torch.Tensor:
        """Keep the most recently used panorama cached within each data-loader worker."""
        image = self._image_cache.get(record_index)
        if image is None:
            image = load_image(self.records[record_index].image_path, self.input_channels)
            self._image_cache[record_index] = image
            if len(self._image_cache) > 1:
                self._image_cache.popitem(last=False)
        else:
            self._image_cache.move_to_end(record_index)
        return image

    def plan(self, record_index: int, initial_offset: int = 0) -> SlidingWindowPlan:
        image = self._load(record_index)
        return SlidingWindowPlan(image.shape[-1], self.window_width, self.stride, initial_offset)

    def build(self, record_index: int, window_index: int, initial_offset: int = 0) -> WindowSample:
        record = self.records[record_index]
        image = self._load(record_index)
        plan = SlidingWindowPlan(image.shape[-1], self.window_width, self.stride, initial_offset)
        start, end = plan.interval(window_index)
        crop = image[:, : self.crop_height, start:end]
        valid_width = crop.shape[-1]
        if crop.shape[-2] < self.crop_height or crop.shape[-1] < self.window_width:
            crop = F.pad(crop, (0, self.window_width - crop.shape[-1],
                                0, self.crop_height - crop.shape[-2]))

        global_boxes = torch.tensor([item.box for item in record.annotations], dtype=torch.float32)
        labels = torch.tensor([item.class_id for item in record.annotations], dtype=torch.long)
        object_ids = torch.tensor([item.object_id for item in record.annotations], dtype=torch.long)
        if global_boxes.numel() == 0:
            global_boxes = torch.empty((0, 4), dtype=torch.float32)
        visible = ((global_boxes[:, 2] > start) & (global_boxes[:, 0] < end))
        local_boxes = clip_xyxy(global_to_local_xyxy(global_boxes[visible], start),
                                self.window_width, self.crop_height)
        areas = (local_boxes[:, 2] - local_boxes[:, 0]) * (local_boxes[:, 3] - local_boxes[:, 1])
        retained = areas > 0

        all_boxes = np.asarray(
            [item.box for item in record.annotations], dtype=np.float32
        ).reshape(-1, 4)
        previous_start = plan.starts[window_index - 1] if window_index > 0 else None
        novelty = torch.from_numpy(novelty_targets(
            all_boxes, start, previous_start, self.window_width
        ))[visible][retained]
        return WindowSample(
            image=crop,
            boxes=local_boxes[retained],
            labels=labels[visible][retained],
            object_ids=object_ids[visible][retained],
            novelty=novelty,
            start_x=start,
            panorama_id=record.panorama_id,
            valid_width=valid_width,
        )


class LocalWindowDataset(Dataset[WindowSample]):
    """All independent windows used for Stage 1 detector pre-training."""

    def __init__(self, records: Sequence[PanoramaRecord], crop_height: int, window_width: int,
                 stride: int, input_channels: int = 3) -> None:
        self.factory = _WindowFactory(records, crop_height, window_width, stride, input_channels)
        self.index: list[tuple[int, int]] = []
        for record_index in range(len(records)):
            window_count = len(self.factory.plan(record_index).starts)
            self.index.extend(
                (record_index, window_index) for window_index in range(window_count)
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> WindowSample:
        record_index, window_index = self.index[index]
        return self.factory.build(record_index, window_index)


class PanoramaSequenceDataset(Dataset[list[WindowSample]]):
    """Random consecutive subsequences with one consistent axial displacement."""

    def __init__(self, records: Sequence[PanoramaRecord], crop_height: int, window_width: int,
                 stride: int, sequence_length: int, input_channels: int = 3,
                 random_axial_shift: bool = True, noise_std: float = 0.01) -> None:
        self.factory = _WindowFactory(records, crop_height, window_width, stride, input_channels)
        self.sequence_length = sequence_length
        self.random_axial_shift = random_axial_shift
        self.augmenter = SequenceAugmenter(noise_std=noise_std)

    def __len__(self) -> int:
        return len(self.factory.records)

    def __getitem__(self, record_index: int) -> list[WindowSample]:
        offset = random.randrange(self.factory.stride) if self.random_axial_shift else 0
        plan = self.factory.plan(record_index, offset)
        max_first = max(0, len(plan.starts) - self.sequence_length)
        first = random.randint(0, max_first)
        indices = range(first, min(first + self.sequence_length, len(plan.starts)))
        samples = [self.factory.build(record_index, index, offset) for index in indices]
        images, boxes = self.augmenter(
            [sample.image for sample in samples], [sample.boxes for sample in samples]
        )
        for sample, image, adjusted_boxes in zip(samples, images, boxes, strict=True):
            sample.image = image
            sample.boxes = adjusted_boxes
        return samples


def collate_windows(samples: list[WindowSample]) -> tuple[
        torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.stack([sample.image for sample in samples])
    targets = [
        {
            "boxes": sample.boxes,
            "labels": sample.labels,
            "object_ids": sample.object_ids,
            "novelty": sample.novelty,
            "start_x": torch.tensor(sample.start_x),
        }
        for sample in samples
    ]
    return images, targets


def collate_sequences(batch: list[list[WindowSample]]) -> list[list[WindowSample]]:
    """Keep variable-length panorama sequences separate for recurrent iteration."""
    return batch
