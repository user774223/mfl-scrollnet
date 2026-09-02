"""Manifest parsing and normalized image loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .types import Annotation, PanoramaRecord


def load_manifest(path: str | Path, image_root: str | Path) -> list[PanoramaRecord]:
    manifest_path = Path(path)
    root = Path(image_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("panoramas"), list):
        raise ValueError("Manifest must contain a 'panoramas' list")
    records: list[PanoramaRecord] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(payload["panoramas"]):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Panorama entry {index} must be an object")
        missing = {"id", "image"} - entry.keys()
        if missing:
            raise ValueError(f"Panorama entry {index} is missing fields: {sorted(missing)}")
        panorama_id = str(entry["id"])
        if panorama_id in seen_ids:
            raise ValueError(f"Duplicate panorama ID: {panorama_id!r}")
        seen_ids.add(panorama_id)
        annotations_list = entry.get("annotations", [])
        if not isinstance(annotations_list, list):
            raise ValueError(f"Annotations for {panorama_id!r} must be a list")
        try:
            annotations = tuple(
                Annotation(
                    box=tuple(float(value) for value in item["bbox_xyxy"]),
                    class_id=int(item["class_id"]),
                    object_id=int(item["object_id"]),
                )
                for item in annotations_list
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid annotation in panorama {panorama_id!r}: {error}") from error
        image_path = Path(entry["image"])
        records.append(PanoramaRecord(
            image_path=image_path if image_path.is_absolute() else root / image_path,
            panorama_id=panorama_id,
            annotations=annotations,
        ))
    return records


def load_image(path: str | Path, channels: int = 3) -> torch.Tensor:
    if channels not in (1, 3):
        raise ValueError("Image loader supports one or three channels")
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array.copy()).permute(2, 0, 1)
