"""Sequence-consistent MFL augmentations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SequenceAugmenter:
    noise_std: float = 0.01
    horizontal_reflection_probability: float = 0.0

    def __call__(self, images: list[torch.Tensor], boxes: list[torch.Tensor],
                 generator: torch.Generator | None = None) -> tuple[
                     list[torch.Tensor], list[torch.Tensor]]:
        if not images:
            return images, boxes
        augmented = [image.clone() for image in images]
        adjusted_boxes = [item.clone() for item in boxes]
        if self.noise_std > 0:
            augmented = [
                (image + torch.randn(image.shape, generator=generator, device=image.device,
                                     dtype=image.dtype) * self.noise_std).clamp(0, 1)
                for image in augmented
            ]
        if self.horizontal_reflection_probability > 0:
            draw = torch.rand((), generator=generator).item()
            if draw < self.horizontal_reflection_probability:
                for index, image in enumerate(augmented):
                    width = image.shape[-1]
                    augmented[index] = image.flip(-1)
                    x1 = adjusted_boxes[index][:, 0].clone()
                    x2 = adjusted_boxes[index][:, 2].clone()
                    adjusted_boxes[index][:, 0] = width - x2
                    adjusted_boxes[index][:, 2] = width - x1
        return augmented, adjusted_boxes
