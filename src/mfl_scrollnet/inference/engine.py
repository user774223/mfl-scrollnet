"""Memory-bounded sequential panorama inference."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from torch.nn import functional as F

from mfl_scrollnet.config import ExperimentConfig
from mfl_scrollnet.data.windowing import SlidingWindowPlan, local_to_global_xyxy
from mfl_scrollnet.data.io import load_image
from mfl_scrollnet.models import MFLScrollNet
from mfl_scrollnet.utils.serialization import save_json
from mfl_scrollnet.utils.tensors import resolve_device

from .association import GlobalAssociator, GlobalDetection


class PanoramaInferenceEngine:
    def __init__(self, model: MFLScrollNet, config: ExperimentConfig,
                 device: torch.device | str = "cpu") -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.config = config
        self.associator = GlobalAssociator(config.association)

    def _windows(self, panorama: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
        plan = SlidingWindowPlan(panorama.shape[-1], self.config.data.window_width,
                                 self.config.data.stride)
        for start in plan.starts:
            crop = panorama[:, :self.config.data.crop_height,
                            start:start + self.config.data.window_width]
            crop = F.pad(crop, (0, self.config.data.window_width - crop.shape[-1],
                                0, self.config.data.crop_height - crop.shape[-2]))
            yield start, crop

    @torch.inference_mode()
    def predict(self, panorama: torch.Tensor,
                progress: Callable[[int, int], None] | None = None) -> list[GlobalDetection]:
        if panorama.ndim != 3:
            raise ValueError("A panorama must have shape [channels, height, width]")
        if panorama.shape[0] != self.config.data.input_channels:
            raise ValueError(
                f"Expected {self.config.data.input_channels} channel(s), got {panorama.shape[0]}"
            )
        if not panorama.is_floating_point():
            raise TypeError("Panorama tensor must use a floating-point dtype")
        self.associator.reset()
        hidden = self.model.initial_state(1, self.device)
        plan = SlidingWindowPlan(panorama.shape[-1], self.config.data.window_width,
                                 self.config.data.stride)
        total_windows = len(plan.starts)
        for window_index, (start, crop) in enumerate(self._windows(panorama)):
            outputs, hidden, _ = self.model.forward_window(
                crop.unsqueeze(0).to(self.device), hidden
            )
            output = outputs[0]
            global_boxes = local_to_global_xyxy(output.boxes, start)
            self.associator.update(global_boxes, output.scores, output.labels,
                                   output.novelty, window_index)
            if progress is not None:
                progress(window_index + 1, total_windows)
        return self.associator.finish()

    def predict_file(self, image_path: str | Path,
                     output_path: str | Path | None = None) -> list[GlobalDetection]:
        """Load an image, run sequential inference, and optionally save portable JSON."""
        detections = self.predict(load_image(image_path, self.config.data.input_channels))
        if output_path is not None:
            save_json(output_path, [item.to_dict() for item in detections])
        return detections
