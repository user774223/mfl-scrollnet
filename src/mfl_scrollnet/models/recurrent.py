"""Object bottleneck, recurrent aggregation, and context-conditioned heads."""

from __future__ import annotations

import torch
from torch import nn


class ObjectContextModule(nn.Module):
    def __init__(self, feature_channels: int, roi_size: int, embedding_dim: int,
                 hidden_dim: int, context_dim: int, num_classes: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_channels * roi_size * roi_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(embedding_dim, hidden_dim)
        self.context = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )
        self.box_head = nn.Linear(context_dim, 4)
        self.objectness_head = nn.Linear(context_dim, 1)
        self.class_head = nn.Linear(context_dim, num_classes)
        self.novelty_head = nn.Linear(context_dim, 1)
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def forward(self, roi_features: torch.Tensor, counts: list[int],
                previous: torch.Tensor) -> tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.project(roi_features)
        descriptors: list[torch.Tensor] = []
        cursor = 0
        for count in counts:
            descriptors.append(
                embeddings[cursor:cursor + count].mean(dim=0)
                if count else embeddings.new_zeros(embeddings.shape[-1])
            )
            cursor += count
        descriptor_batch = torch.stack(descriptors)
        hidden = self.gru(descriptor_batch, previous)
        repeated_hidden = torch.cat([
            hidden[index:index + 1].expand(count, -1)
            for index, count in enumerate(counts) if count
        ], dim=0) if sum(counts) else hidden.new_empty((0, hidden.shape[-1]))
        context = self.context(torch.cat((embeddings, repeated_hidden), dim=-1))
        return (
            hidden,
            self.box_head(context),
            self.objectness_head(context).squeeze(-1).sigmoid(),
            self.class_head(context).softmax(dim=-1),
            self.novelty_head(context).squeeze(-1).sigmoid(),
        )
