"""Stable JSON conversion and atomic persistence for experiment artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_jsonable(value: Any) -> Any:
    """Convert tensors, arrays, paths, dataclasses, and nested containers to JSON values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write UTF-8 JSON so interrupted inference cannot leave a partial result."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
