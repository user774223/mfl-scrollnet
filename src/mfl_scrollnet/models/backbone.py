from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvNormAct, DownStage


class DarknetFPN(nn.Module):
    def __init__(self, input_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = ConvNormAct(input_channels, c, 3)
        self.stage1 = DownStage(c, c * 2, 1)
        self.stage2 = DownStage(c * 2, c * 4, 2)
        self.stage3 = DownStage(c * 4, c * 8, 3)
        self.stage4 = DownStage(c * 8, c * 16, 3)
        self.stage5 = DownStage(c * 16, c * 32, 2)
        self.lateral5 = ConvNormAct(c * 32, c * 16, 1)
        self.lateral4 = ConvNormAct(c * 16, c * 16, 1)
        self.merge4 = ConvNormAct(c * 32, c * 16, 3)
        self.reduce4 = ConvNormAct(c * 16, c * 8, 1)
        self.lateral3 = ConvNormAct(c * 8, c * 8, 1)
        self.merge3 = ConvNormAct(c * 16, c * 8, 3)
        self.out_channels = (c * 8, c * 16, c * 16)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.stage2(self.stage1(self.stem(inputs)))
        c3 = self.stage3(value)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p5 = self.lateral5(c5)
        p4 = self.merge4(torch.cat((F.interpolate(p5, size=c4.shape[-2:], mode="nearest"),
                                    self.lateral4(c4)), dim=1))
        p3 = self.merge3(torch.cat((F.interpolate(self.reduce4(p4), size=c3.shape[-2:],
                                                  mode="nearest"), self.lateral3(c3)), dim=1))
        return p3, p4, p5

