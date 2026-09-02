"""General utilities exposed as a compact public API."""

from .boxes import box_area, box_iou, clip_boxes, valid_box_mask
from .seed import seed_everything, seed_worker, temporary_seed
from .serialization import load_json, save_json, to_jsonable
from .tensors import ensure_finite, move_to_device, parameter_count, resolve_device
from .visualization import (
    plot_class_metrics,
    plot_confusion_matrix,
    plot_detections,
    plot_precision_recall,
    plot_score_distribution,
    plot_training_history,
    save_figure,
    smooth_curve,
)

__all__ = [
    "box_area", "box_iou", "clip_boxes", "valid_box_mask",
    "seed_everything", "seed_worker", "temporary_seed",
    "load_json", "save_json", "to_jsonable",
    "ensure_finite", "move_to_device", "parameter_count", "resolve_device",
    "plot_class_metrics", "plot_confusion_matrix", "plot_detections",
    "plot_precision_recall", "plot_score_distribution", "plot_training_history",
    "save_figure", "smooth_curve",
]
