"""Reproducibility controls."""

import random
from contextlib import contextmanager
from collections.abc import Iterator

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch and optionally request deterministic kernels."""
    if seed < 0:
        raise ValueError("Seed cannot be negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """Initialize a data-loader worker from PyTorch's assigned initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Temporarily isolate Python, NumPy, and CPU PyTorch random state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
