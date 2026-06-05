"""Utilities for Hydra-driven training/data-generation scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

__all__ = [
    "ensure_experiment_dir",
    "attach_file_logger",
    "log_config",
    "resolve_device",
]


def ensure_experiment_dir(output_dir: str | Path) -> Path:
    """Create (if needed) and return the experiment directory."""
    exp_dir = Path(output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def attach_file_logger(config, exp_dir: Path) -> Optional[logging.Handler]:
    """Attach a file handler to the root logger if enabled in config."""
    if not config.logging.get("save_to_file", True):
        return None

    log_file = exp_dir / config.logging.get("file_name", "training.log")
    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers when scripts are re-run in the same process.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_file):
            return handler

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    return file_handler


def log_config(logger: logging.Logger, config, title: str) -> None:
    """Standardised header + config logging."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("\n%s", OmegaConf.to_yaml(config))
    logger.info("=" * 60)


def resolve_device(preferred: str, logger: logging.Logger) -> str:
    """Pick CUDA when requested and available, otherwise fall back to CPU."""
    if preferred == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA not available, falling back to CPU")
        return "cpu"
    return preferred
