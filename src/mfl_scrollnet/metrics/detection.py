from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mfl_scrollnet.utils.boxes import box_iou


@dataclass(slots=True)
class ImageDetections:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.boxes.ndim != 2 or self.boxes.shape[-1] != 4:
            raise ValueError("Detection boxes must have shape [N, 4]")
        count = self.boxes.shape[0]
        if self.scores.ndim != 1 or self.labels.ndim != 1:
            raise ValueError("Detection scores and labels must have shape [N]")
        if self.scores.shape[0] != count or self.labels.shape[0] != count:
            raise ValueError("Detection tensors must contain the same number of entries")


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    recall_points = np.linspace(0, 1, 101)
    interpolated = [precision[recall >= point].max() if np.any(recall >= point) else 0.0
                    for point in recall_points]
    return float(np.mean(interpolated))


class DetectionEvaluator:
    def __init__(self, num_classes: int) -> None:
        if num_classes <= 0:
            raise ValueError("Number of classes must be positive")
        self.num_classes = num_classes
        self.predictions: list[ImageDetections] = []
        self.targets: list[ImageDetections] = []

    def add(self, prediction: ImageDetections, target: ImageDetections) -> None:
        if prediction.labels.numel() and (
            int(prediction.labels.min()) < 0 or int(prediction.labels.max()) >= self.num_classes
        ):
            raise ValueError("Prediction contains a class outside the configured range")
        if target.labels.numel() and (
            int(target.labels.min()) < 0 or int(target.labels.max()) >= self.num_classes
        ):
            raise ValueError("Target contains a class outside the configured range")
        self.predictions.append(prediction)
        self.targets.append(target)

    def reset(self) -> None:
        self.predictions.clear()
        self.targets.clear()

    def _class_results(self, class_id: int, iou_threshold: float,
                       score_threshold: float | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        candidates: list[tuple[float, int, int]] = []
        ground_truth_count = 0
        matched: list[torch.Tensor] = []
        for image_index, (prediction, target) in enumerate(zip(
                self.predictions, self.targets, strict=True)):
            prediction_indices = torch.where(prediction.labels == class_id)[0]
            if score_threshold is not None:
                prediction_indices = prediction_indices[
                    prediction.scores[prediction_indices] >= score_threshold
                ]
            target_indices = torch.where(target.labels == class_id)[0]
            ground_truth_count += target_indices.numel()
            matched.append(torch.zeros(target_indices.numel(), dtype=torch.bool))
            candidates.extend((float(prediction.scores[index]), image_index, int(index))
                              for index in prediction_indices)
        candidates.sort(reverse=True)
        true_positives = np.zeros(len(candidates), dtype=np.float32)
        false_positives = np.zeros(len(candidates), dtype=np.float32)
        for rank, (_, image_index, prediction_index) in enumerate(candidates):
            prediction = self.predictions[image_index]
            target = self.targets[image_index]
            target_indices = torch.where(target.labels == class_id)[0]
            if target_indices.numel() == 0:
                false_positives[rank] = 1
                continue
            overlaps = box_iou(prediction.boxes[prediction_index].view(1, 4),
                               target.boxes[target_indices])[0]
            best = int(overlaps.argmax())
            if float(overlaps[best]) >= iou_threshold and not matched[image_index][best]:
                true_positives[rank] = 1
                matched[image_index][best] = True
            else:
                false_positives[rank] = 1
        return true_positives, false_positives, ground_truth_count

    def mean_ap(self, thresholds: tuple[float, ...]) -> float:
        values = []
        for threshold in thresholds:
            for class_id in range(self.num_classes):
                tp, fp, gt_count = self._class_results(class_id, threshold)
                if gt_count == 0:
                    continue
                recall = np.cumsum(tp) / gt_count
                precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1e-12)
                values.append(_average_precision(recall, precision))
        return float(np.mean(values)) if values else 0.0

    def operating_point(self, score_threshold: float = 0.25,
                        iou_threshold: float = 0.5) -> dict[str, float]:
        per_class = []
        total_tp = total_fp = total_fn = 0.0
        for class_id in range(self.num_classes):
            tp_values, fp_values, gt_count = self._class_results(
                class_id, iou_threshold, score_threshold
            )
            tp, fp = float(tp_values.sum()), float(fp_values.sum())
            fn = max(gt_count - tp, 0)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            per_class.append((precision, recall, f1))
            total_tp += tp
            total_fp += fp
            total_fn += fn
        macro = np.asarray(per_class).mean(axis=0)
        micro_precision = total_tp / max(total_tp + total_fp, 1)
        micro_recall = total_tp / max(total_tp + total_fn, 1)
        micro_f1 = 2 * micro_precision * micro_recall / max(
            micro_precision + micro_recall, 1e-12
        )
        return {
            "precision_macro": float(macro[0]), "recall_macro": float(macro[1]),
            "f1_macro": float(macro[2]), "precision_micro": micro_precision,
            "recall_micro": micro_recall, "f1_micro": micro_f1,
        }

    def per_class(self, score_threshold: float = 0.25,
                  iou_threshold: float = 0.5) -> list[dict[str, float | int]]:
        """Return inspectable class-level counts and metrics at one operating point."""
        result: list[dict[str, float | int]] = []
        for class_id in range(self.num_classes):
            tp_values, fp_values, target_count = self._class_results(
                class_id, iou_threshold, score_threshold
            )
            tp, fp = int(tp_values.sum()), int(fp_values.sum())
            fn = max(target_count - tp, 0)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            result.append({
                "class_id": class_id, "true_positives": tp, "false_positives": fp,
                "false_negatives": fn, "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            })
        return result

    def summary(self) -> dict[str, float]:
        values = {
            "mAP50": self.mean_ap((0.5,)),
            "mAP50_95": self.mean_ap(tuple(float(value) for value in np.arange(0.5, 1.0, 0.05))),
        }
        values.update(self.operating_point())
        return values
