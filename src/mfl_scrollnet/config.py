from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    manifest: str = "data/manifest.json"
    image_root: str = "data/images"
    input_channels: int = 3
    crop_height: int = 1080
    window_width: int = 128
    stride: int = 112
    sequence_length: int = 16
    class_names: tuple[str, ...] = (
        "MTL", "CRC", "GWA", "SWA", "BND", "SLE",
        "BRN", "TEE", "CAS", "VAL", "ESP", "FLA",
    )
    gaussian_noise_std: float = 0.01
    random_axial_shift: bool = True


@dataclass(slots=True)
class ModelConfig:
    base_channels: int = 32
    embedding_dim: int = 256
    hidden_dim: int = 256
    context_dim: int = 256
    roi_size: int = 5
    anchors: tuple[tuple[float, float], ...] = (
        (8.0, 8.0), (16.0, 12.0), (24.0, 24.0),
        (36.0, 28.0), (56.0, 40.0), (80.0, 64.0),
        (100.0, 90.0), (120.0, 160.0), (128.0, 320.0),
    )
    score_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    max_detections_per_window: int = 100


@dataclass(slots=True)
class AssociationConfig:
    iou_threshold: float = 0.3
    novelty_threshold: float = 0.5
    max_inactive_windows: int = 1
    confidence_weighted_fusion: bool = True


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    workers: int = 4
    epochs_local: int = 100
    epochs_sequential: int = 100
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    restart_period: int = 10
    restart_multiplier: int = 2
    box_weight: float = 5.0
    objectness_weight: float = 1.0
    class_weight: float = 1.0
    novelty_weight: float = 1.0
    gradient_clip_norm: float = 5.0
    amp: bool = True
    freeze_backbone_in_stage2: bool = True
    seed: int = 42


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.data.input_channels not in (1, 3):
            raise ValueError("Input channels must be 1 or 3")
        if self.data.crop_height <= 0:
            raise ValueError("Crop height must be positive")
        if self.data.window_width <= 0 or self.data.stride <= 0:
            raise ValueError("Window width and stride must be positive")
        if self.data.stride > self.data.window_width:
            raise ValueError("Stride cannot exceed window width")
        if self.data.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")
        if not self.data.class_names:
            raise ValueError("At least one class is required")
        if len(self.model.anchors) % 3 != 0:
            raise ValueError("The anchor count must be divisible by three detection scales")
        if any(width <= 0 or height <= 0 for width, height in self.model.anchors):
            raise ValueError("Anchor dimensions must be positive")
        for name in ("base_channels", "embedding_dim", "hidden_dim", "context_dim", "roi_size"):
            if getattr(self.model, name) <= 0:
                raise ValueError(f"Model value {name!r} must be positive")
        if self.model.max_detections_per_window <= 0:
            raise ValueError("Maximum detections per window must be positive")
        for value in (
            self.model.score_threshold,
            self.model.nms_iou_threshold,
            self.association.iou_threshold,
            self.association.novelty_threshold,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Probability and IoU thresholds must be in [0, 1]")
        if self.association.max_inactive_windows < 0:
            raise ValueError("Inactive-window lifetime cannot be negative")
        for name in ("batch_size", "epochs_local", "epochs_sequential", "restart_period"):
            if getattr(self.training, name) <= 0:
                raise ValueError(f"Training value {name!r} must be positive")
        if self.training.workers < 0 or self.training.weight_decay < 0:
            raise ValueError("Workers and weight decay cannot be negative")
        if self.training.learning_rate <= 0 or self.training.gradient_clip_norm <= 0:
            raise ValueError("Learning rate and gradient clip norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_resolved_paths(self, config_path: str | Path) -> ExperimentConfig:
        """Return a copy whose dataset paths are relative to the config directory."""
        base = Path(config_path).expanduser().resolve().parent
        values = self.to_dict()
        for key in ("manifest", "image_root"):
            path = Path(values["data"][key]).expanduser()
            values["data"][key] = str(path if path.is_absolute() else base / path)
        return config_from_dict(values)


def _construct(cls: type, values: dict[str, Any]) -> Any:
    values = dict(values)
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    if cls is DataConfig and "class_names" in values:
        values["class_names"] = tuple(values["class_names"])
    if cls is ModelConfig and "anchors" in values:
        values["anchors"] = tuple(tuple(pair) for pair in values["anchors"])
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build and validate a configuration from a plain mapping."""
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping")
    allowed = {"data", "model", "association", "training"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for section in allowed & raw.keys():
        if not isinstance(raw[section], dict):
            raise TypeError(f"Configuration section {section!r} must be a mapping")
    config = ExperimentConfig(
        data=_construct(DataConfig, raw.get("data", {})),
        model=_construct(ModelConfig, raw.get("model", {})),
        association=_construct(AssociationConfig, raw.get("association", {})),
        training=_construct(TrainingConfig, raw.get("training", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML configuration and reject unknown top-level sections."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config_from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Persist a validated configuration as human-readable YAML."""
    config.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
