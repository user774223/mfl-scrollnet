from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


DEFAULT_FIGURE_SIZE = (9.0, 5.0)


def _numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _figure(ax: Axes | None, figsize: tuple[float, float] = DEFAULT_FIGURE_SIZE) -> tuple[
        Figure, Axes]:
    if ax is None:
        figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
        return figure, axes
    return ax.figure, ax


def _validate_series(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = _numpy(values, float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def smooth_curve(values: Sequence[float] | np.ndarray, window: int = 1) -> np.ndarray:
    """Return a centered moving average while preserving the original length."""
    series = _validate_series(values, "values")
    if window <= 0:
        raise ValueError("Smoothing window must be positive")
    if window == 1 or series.size == 1:
        return series.copy()
    window = min(window, series.size)
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(series, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_training_history(
    history: Sequence[Any],
    metrics: Sequence[str] | None = None,
    smooth_window: int = 1,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot metrics from EpochResult objects or dictionaries with a ``metrics`` mapping."""
    if not history:
        raise ValueError("Training history cannot be empty")
    epochs: list[int] = []
    rows: list[Mapping[str, float]] = []
    for index, item in enumerate(history):
        if isinstance(item, Mapping):
            epoch = int(item.get("epoch", index))
            row = item.get("metrics", item)
        else:
            epoch = int(getattr(item, "epoch", index))
            row = getattr(item, "metrics", None)
        if not isinstance(row, Mapping):
            raise TypeError("Every history item must provide a metrics mapping")
        epochs.append(epoch)
        rows.append(row)
    selected = list(metrics) if metrics is not None else sorted(
        set.intersection(*(set(row) for row in rows))
    )
    selected = [name for name in selected if name not in {"epoch", "learning_rate"}]
    if not selected:
        raise ValueError("No common metrics found in training history")

    figure, axes = _figure(ax)
    for name in selected:
        values = [float(row[name]) for row in rows]
        axes.plot(epochs, smooth_curve(values, smooth_window), marker="o", label=name)
    axes.set(title="Training history", xlabel="Epoch", ylabel="Metric value")
    axes.grid(alpha=0.25)
    axes.legend()
    return figure, axes


def plot_precision_recall(
    recall: Sequence[float] | np.ndarray,
    precision: Sequence[float] | np.ndarray,
    label: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one precision-recall curve with conventional unit-square bounds."""
    recall_values = _validate_series(recall, "recall")
    precision_values = _validate_series(precision, "precision")
    if recall_values.shape != precision_values.shape:
        raise ValueError("Recall and precision must have the same length")
    if ((recall_values < 0) | (recall_values > 1)).any():
        raise ValueError("Recall values must be in [0, 1]")
    if ((precision_values < 0) | (precision_values > 1)).any():
        raise ValueError("Precision values must be in [0, 1]")

    order = np.argsort(recall_values)
    figure, axes = _figure(ax, (6.0, 6.0))
    axes.step(recall_values[order], precision_values[order], where="post", label=label)
    axes.set(
        title="Precision-recall curve", xlabel="Recall", ylabel="Precision",
        xlim=(0, 1), ylim=(0, 1), aspect="equal",
    )
    axes.grid(alpha=0.25)
    if label:
        axes.legend()
    return figure, axes


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    class_names: Sequence[str],
    normalize: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render a square confusion matrix with optional row normalization."""
    values = _numpy(matrix, float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if values.shape[0] != len(class_names):
        raise ValueError("Class-name count must match the confusion matrix")
    if (values < 0).any() or not np.isfinite(values).all():
        raise ValueError("Confusion matrix must contain finite non-negative values")
    if normalize:
        denominators = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denominators, out=np.zeros_like(values), where=denominators > 0)

    size = max(6.0, min(14.0, 0.65 * len(class_names) + 3.0))
    figure, axes = _figure(ax, (size, size))
    image = axes.imshow(values, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    axes.set(
        title="Normalized confusion matrix" if normalize else "Confusion matrix",
        xlabel="Predicted class", ylabel="True class",
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = float(values.max()) / 2 if values.size else 0
    text_format = ".2f" if normalize else ".0f"
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axes.text(
                column, row, format(values[row, column], text_format),
                ha="center", va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    return figure, axes


def plot_class_metrics(
    per_class: Sequence[Mapping[str, float | int]],
    class_names: Sequence[str] | None = None,
    metrics: Sequence[str] = ("precision", "recall", "f1"),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Compare evaluation metrics across classes as grouped bars."""
    if not per_class:
        raise ValueError("Per-class metrics cannot be empty")
    names = list(class_names) if class_names is not None else [
        str(item.get("class_id", index)) for index, item in enumerate(per_class)
    ]
    if len(names) != len(per_class):
        raise ValueError("Class-name count must match per-class metric rows")
    if not metrics:
        raise ValueError("At least one metric is required")
    values = np.asarray(
        [[float(item[name]) for item in per_class] for name in metrics], dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("Class metrics contain non-finite values")

    width = 0.8 / len(metrics)
    positions = np.arange(len(per_class))
    figure, axes = _figure(ax, (max(9.0, len(names) * 0.65), 5.0))
    for index, name in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        axes.bar(positions + offset, values[index], width=width, label=name)
    axes.set(
        title="Metrics by class", xlabel="Class", ylabel="Score",
        xticks=positions, xticklabels=names, ylim=(0, max(1.0, float(values.max()) * 1.05)),
    )
    axes.grid(axis="y", alpha=0.25)
    axes.legend()
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    return figure, axes


def plot_score_distribution(
    scores: Sequence[float] | np.ndarray | torch.Tensor,
    labels: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    class_names: Sequence[str] | None = None,
    bins: int = 20,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot confidence histograms globally or split by class label."""
    score_values = _validate_series(scores, "scores")
    if ((score_values < 0) | (score_values > 1)).any():
        raise ValueError("Scores must be in [0, 1]")
    if bins <= 0:
        raise ValueError("Bin count must be positive")
    figure, axes = _figure(ax)
    if labels is None:
        axes.hist(score_values, bins=bins, range=(0, 1), alpha=0.8)
    else:
        label_values = _numpy(labels, int).reshape(-1)
        if label_values.shape != score_values.shape:
            raise ValueError("Labels and scores must have the same length")
        for class_id in np.unique(label_values):
            if class_id < 0:
                raise ValueError("Class labels cannot be negative")
            name = (
                class_names[int(class_id)]
                if class_names is not None and class_id < len(class_names)
                else str(class_id)
            )
            axes.hist(
                score_values[label_values == class_id], bins=bins, range=(0, 1),
                histtype="step", linewidth=2, label=name,
            )
        axes.legend()
    axes.set(
        title="Detection score distribution", xlabel="Confidence", ylabel="Detections",
        xlim=(0, 1),
    )
    axes.grid(axis="y", alpha=0.25)
    return figure, axes


def plot_detections(
    image: np.ndarray | torch.Tensor,
    boxes: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor | None = None,
    class_names: Sequence[str] | None = None,
    score_threshold: float = 0.0,
    max_detections: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Draw XYXY detections over CHW or HWC image data."""
    image_values = _numpy(image)
    if image_values.ndim == 3 and image_values.shape[0] in (1, 3, 4):
        image_values = np.moveaxis(image_values, 0, -1)
    if image_values.ndim == 3 and image_values.shape[-1] == 1:
        image_values = image_values[..., 0]
    if image_values.ndim not in (2, 3):
        raise ValueError("Image must have HW, CHW, or HWC shape")
    box_values = _numpy(boxes, float).reshape(-1, 4)
    label_values = _numpy(labels, int).reshape(-1)
    if label_values.shape[0] != box_values.shape[0]:
        raise ValueError("Labels and boxes must contain the same number of entries")
    score_values = (
        np.ones(box_values.shape[0], dtype=float)
        if scores is None else _numpy(scores, float).reshape(-1)
    )
    if score_values.shape[0] != box_values.shape[0]:
        raise ValueError("Scores and boxes must contain the same number of entries")
    if not 0 <= score_threshold <= 1:
        raise ValueError("Score threshold must be in [0, 1]")
    if max_detections is not None and max_detections <= 0:
        raise ValueError("Maximum detections must be positive")

    selected = np.flatnonzero(score_values >= score_threshold)
    selected = selected[np.argsort(score_values[selected])[::-1]]
    if max_detections is not None:
        selected = selected[:max_detections]
    figure, axes = _figure(ax, (12.0, 6.0))
    axes.imshow(np.clip(image_values, 0, 1) if image_values.dtype.kind == "f" else image_values,
                cmap="gray" if image_values.ndim == 2 else None)
    colors = plt.get_cmap("tab20")
    for index in selected:
        x1, y1, x2, y2 = box_values[index]
        if not np.isfinite(box_values[index]).all() or x2 <= x1 or y2 <= y1:
            continue
        class_id = int(label_values[index])
        color = colors(class_id % colors.N)
        axes.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
        ))
        class_name = (
            class_names[class_id]
            if class_names is not None and 0 <= class_id < len(class_names)
            else str(class_id)
        )
        caption = class_name if scores is None else f"{class_name} {score_values[index]:.2f}"
        axes.text(
            x1, max(y1 - 2, 0), caption, color="white", fontsize=9, va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
    axes.set_title(f"Detections ({len(selected)})")
    axes.axis("off")
    return figure, axes


def save_figure(
    figure: Figure,
    path: str | Path,
    dpi: int = 160,
    transparent: bool = False,
    close: bool = False,
) -> Path:
    """Atomically save a figure and optionally release its Matplotlib resources."""
    if dpi <= 0:
        raise ValueError("DPI must be positive")
    destination = Path(path)
    if not destination.suffix:
        destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.stem}.tmp{destination.suffix}"
    )
    try:
        figure.savefig(temporary, dpi=dpi, transparent=transparent, bbox_inches="tight")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        if close:
            plt.close(figure)
    return destination
