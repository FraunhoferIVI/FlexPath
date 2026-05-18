"""Small shared utilities used across scripts and modules."""

from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch

__all__ = ["seed_everything", "count_parameters"]


def seed_everything(seed: int = 24022022) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA when available).

    Args:
        seed: The seed to apply.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = True) -> int:
    """Count parameters in a model.

    Args:
        model: A PyTorch module.
        trainable_only: When True (default), only parameters with ``requires_grad``
            are counted.
    """
    if trainable_only:
        parameters: Iterable[torch.nn.Parameter] = (p for p in model.parameters() if p.requires_grad)
    else:
        parameters = model.parameters()
    return sum(p.numel() for p in parameters)
