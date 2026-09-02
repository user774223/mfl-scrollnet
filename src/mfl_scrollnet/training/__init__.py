"""Target encoding, losses, and two-stage optimization."""

from .losses import DetectionLoss, SequentialLoss
from .trainer import EpochResult, Trainer

__all__ = ["DetectionLoss", "SequentialLoss", "EpochResult", "Trainer"]
