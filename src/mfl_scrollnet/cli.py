from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .data import (
    LocalWindowDataset,
    PanoramaSequenceDataset,
    collate_sequences,
    collate_windows,
    load_manifest,
)
from .inference import PanoramaInferenceEngine
from .metrics import DetectionEvaluator, ImageDetections
from .models import MFLScrollNet
from .training import EpochResult, Trainer
from .training.checkpoint import load_checkpoint
from .utils.seed import seed_everything, seed_worker
from .utils.serialization import load_json, save_json
from .utils.tensors import parameter_count, resolve_device


def _records(config: ExperimentConfig):
    return load_manifest(config.data.manifest, config.data.image_root)


def _loader_options(config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    return {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }


def _report_epoch(result: EpochResult) -> None:
    payload = {
        "epoch": result.epoch,
        "stage": result.stage,
        "learning_rate": result.learning_rate,
        **result.metrics,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _build_model(config: ExperimentConfig, checkpoint: str | Path | None = None) -> MFLScrollNet:
    model = MFLScrollNet(config)
    if checkpoint is not None:
        load_checkpoint(checkpoint, model)
    return model


def train_local(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = resolve_device(args.device)
    seed_everything(config.training.seed)
    dataset = LocalWindowDataset(
        _records(config), config.data.crop_height, config.data.window_width,
        config.data.stride, config.data.input_channels,
    )
    loader = DataLoader(
        dataset, shuffle=True, collate_fn=collate_windows,
        **_loader_options(config, device),
    )
    model = _build_model(config, args.resume if args.weights_only else None)
    trainer = Trainer(model, config, device)
    if args.resume and not args.weights_only:
        trainer.resume(args.resume)
    print(f"device={device} parameters={parameter_count(model):,} windows={len(dataset):,}")
    trainer.fit_local(loader, args.epochs, args.output, _report_epoch)


def train_sequential(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = resolve_device(args.device)
    seed_everything(config.training.seed)
    dataset = PanoramaSequenceDataset(
        _records(config), config.data.crop_height, config.data.window_width,
        config.data.stride, config.data.sequence_length, config.data.input_channels,
        config.data.random_axial_shift, config.data.gaussian_noise_std,
    )
    options = _loader_options(config, device)
    loader = DataLoader(dataset, shuffle=True, collate_fn=collate_sequences, **options)
    model = _build_model(config, args.local_checkpoint)
    if config.training.freeze_backbone_in_stage2:
        model.freeze_backbone()
    trainer = Trainer(model, config, device)
    if args.resume:
        trainer.resume(args.resume)
    print(
        f"device={device} trainable_parameters={parameter_count(model, True):,} "
        f"panoramas={len(dataset):,}"
    )
    trainer.fit_sequential(loader, args.epochs, args.output, _report_epoch)


def infer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = resolve_device(args.device)
    model = _build_model(config, args.checkpoint)
    engine = PanoramaInferenceEngine(model, config, device)
    detections = engine.predict_file(args.image, args.output)
    print(json.dumps(engine.associator.statistics(), ensure_ascii=False))
    print(f"saved {len(detections)} detection(s) to {args.output}")


def _as_detections(items: list[dict[str, Any]], target: bool = False) -> ImageDetections:
    boxes = torch.tensor([item["bbox_xyxy"] for item in items], dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor([item["class_id"] for item in items], dtype=torch.long)
    scores = torch.ones(len(items)) if target else torch.tensor(
        [item.get("score", 1.0) for item in items], dtype=torch.float32
    )
    return ImageDetections(boxes, scores, labels)


def evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    predictions = load_json(args.predictions)
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must map panorama IDs to detection lists")
    evaluator = DetectionEvaluator(len(config.data.class_names))
    for record in _records(config):
        predicted_items = predictions.get(record.panorama_id, [])
        if not isinstance(predicted_items, list):
            raise ValueError(f"Predictions for {record.panorama_id!r} must be a list")
        evaluator.add(
            _as_detections(predicted_items),
            _as_detections([item.to_dict() for item in record.annotations], target=True),
        )
    summary = evaluator.summary()
    payload = {"summary": summary, "per_class": evaluator.per_class()}
    if args.output:
        save_json(args.output, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfl-scrollnet", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("train-local", help="train the independent-window detector")
    local.add_argument("--config", required=True)
    local.add_argument("--output", required=True)
    local.add_argument("--resume")
    local.add_argument("--weights-only", action="store_true")
    local.add_argument("--epochs", type=int)
    local.add_argument("--device", default="auto")
    local.set_defaults(handler=train_local)

    sequential = subparsers.add_parser("train-sequential", help="train recurrent context heads")
    sequential.add_argument("--config", required=True)
    sequential.add_argument("--local-checkpoint", required=True)
    sequential.add_argument("--resume")
    sequential.add_argument("--output", required=True)
    sequential.add_argument("--epochs", type=int)
    sequential.add_argument("--device", default="auto")
    sequential.set_defaults(handler=train_sequential)

    prediction = subparsers.add_parser("infer", help="run panorama inference")
    prediction.add_argument("--config", required=True)
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="auto")
    prediction.set_defaults(handler=infer)

    evaluation = subparsers.add_parser("evaluate", help="evaluate global predictions")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--output")
    evaluation.set_defaults(handler=evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
