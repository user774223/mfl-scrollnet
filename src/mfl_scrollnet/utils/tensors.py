from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve ``auto`` to CUDA when available and validate explicit CUDA requests."""
    if requested is None or str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_to_device(value: Any, device: torch.device | str, non_blocking: bool = False) -> Any:
    """Recursively move tensors while preserving common Python containers."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)((key, move_to_device(item, device, non_blocking))
                           for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    return value


def parameter_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters, optionally excluding frozen parameters."""
    parameters = (item for item in module.parameters() if item.requires_grad) \
        if trainable_only else module.parameters()
    return sum(item.numel() for item in parameters)


def ensure_finite(value: torch.Tensor, name: str = "tensor") -> None:
    """Raise a diagnostic error before invalid values reach an optimizer or serializer."""
    if not torch.isfinite(value).all():
        invalid = int((~torch.isfinite(value)).sum())
        raise FloatingPointError(f"{name} contains {invalid} non-finite value(s)")
