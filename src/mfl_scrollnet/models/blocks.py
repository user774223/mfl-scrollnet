"""Reusable convolutional building blocks."""

from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.layers = nn.Sequential(
            ConvNormAct(channels, hidden, 1),
            ConvNormAct(hidden, channels, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class DownStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, blocks: int) -> None:
        layers: list[nn.Module] = [ConvNormAct(in_channels, out_channels, 3, 2)]
        layers.extend(ResidualBlock(out_channels) for _ in range(blocks))
        super().__init__(*layers)

