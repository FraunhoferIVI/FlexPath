"""
Checkpoint loading utilities for VSAC.

This module provides functions to load pretrained actor and critic models
using the model registry, enabling dynamic model instantiation from config.

Usage:
    from src.utils.checkpoint_loader import load_pretrained_actor

    actor_model = load_pretrained_actor(
        model_config={"type": "actor_rnt_m", "params": {"num_labels": 1}},
        checkpoint_path="checkpoints/actor_rnt_m.pth.tar",
        device="cuda"
    )
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.utils.model_registry import get_actor_from_config

logger = logging.getLogger(__name__)


def _get_default_device() -> str:
    """Determine the default device based on availability."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ============================================================================
# Auto-loading from experiment directories
# ============================================================================
# These functions enable automatic parameter loading from pretraining
# experiment configs, eliminating the need to manually specify model params.
# ============================================================================


def load_config_from_experiment_dir(exp_dir: Union[str, Path]) -> DictConfig:
    """
    Load config.yaml from an experiment directory.

    Args:
        exp_dir: Path to experiment directory (contains config.yaml)

    Returns:
        Loaded config as DictConfig

    Raises:
        FileNotFoundError: If config.yaml doesn't exist
        ValueError: If config has invalid structure

    Example:
        >>> config = load_config_from_experiment_dir(
        ...     "experiments/pretraining/20251103_121611_actor_actor_rnt_s_42x42"
        ... )
        >>> print(config.model.type)  # "actor_rnt_s"
    """
    from omegaconf import OmegaConf

    exp_dir = Path(exp_dir)
    config_path = exp_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected location: {exp_dir}/config.yaml\n"
            f"This experiment directory may be incomplete or corrupted."
        )

    logger.debug(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    # Validate config structure
    if not hasattr(config, "model"):
        raise ValueError(
            f"Invalid config structure in {config_path}\n"
            f"Expected 'model' key at top level.\n"
            f"Found keys: {list(config.keys())}"
        )

    if not hasattr(config.model, "type"):
        raise ValueError(
            f"Invalid config structure in {config_path}\n"
            f"Expected 'model.type' key.\n"
            f"Found model keys: {list(config.model.keys())}"
        )

    return config


def infer_checkpoint_path(exp_dir: Union[str, Path], use_best: bool = True) -> Path:
    """
    Infer checkpoint path from experiment directory.

    Args:
        exp_dir: Path to experiment directory
        use_best: If True, use best.pth; otherwise use latest.pth

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If checkpoint doesn't exist

    Example:
        >>> checkpoint_path = infer_checkpoint_path(
        ...     "experiments/pretraining/20251103_121611_actor_actor_rnt_s_42x42",
        ...     use_best=True
        ... )
        >>> print(checkpoint_path)  # .../checkpoints/best.pth
    """
    exp_dir = Path(exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_name = "best.pth" if use_best else "latest.pth"
    checkpoint_path = checkpoint_dir / checkpoint_name

    if not checkpoint_path.exists():
        # Try to provide helpful error message
        available_checkpoints = []
        if checkpoint_dir.exists():
            available_checkpoints = list(checkpoint_dir.glob("*.pth"))

        error_msg = (
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Looking for: {checkpoint_name}\n"
            f"In directory: {checkpoint_dir}"
        )
        if available_checkpoints:
            error_msg += "\n\nAvailable checkpoints:\n"
            for ckpt in available_checkpoints:
                error_msg += f"  - {ckpt.name}\n"
        else:
            error_msg += f"\n\nNo checkpoints found in {checkpoint_dir}"

        raise FileNotFoundError(error_msg)

    return checkpoint_path


def load_pretrained_actor_from_experiment(
    exp_dir: Union[str, Path],
    use_best: bool = True,
    device: Optional[str] = None,
    use_weights: bool = True
) -> nn.Module:
    """
    Load pretrained actor by auto-detecting params from experiment config.

    This is the NEW simplified loading method that reads model type and
    params from the experiment's config.yaml, eliminating manual config.

    Args:
        exp_dir: Path to experiment directory containing config.yaml
        use_best: If True, load best.pth; otherwise load latest.pth
        device: Device to load model on (auto-detected if None)

    Returns:
        Loaded actor model ready for use

    Raises:
        FileNotFoundError: If experiment dir, config, or checkpoint not found
        ValueError: If config structure is invalid

    Example:
        >>> actor = load_pretrained_actor_from_experiment(
        ...     exp_dir="experiments/pretraining/20251103_121611_actor_actor_rnt_s_42x42",
        ...     use_best=True,
        ...     device="cuda"
        ... )
        >>> actor.eval()

    See Also:
        load_pretrained_actor() for the OLD method with explicit params
    """
    exp_dir = Path(exp_dir)
    logger.info("Loading actor from experiment directory (NEW auto-loading method)")
    logger.info(f"  Experiment dir: {exp_dir}")
    logger.info(f"  Use best checkpoint: {use_best}")

    # Load config to get model type and params
    config = load_config_from_experiment_dir(exp_dir)

    # Extract model config
    from omegaconf import OmegaConf

    model_config = {
        "type": config.model.type,
        "params": OmegaConf.to_container(config.model.params) if hasattr(config.model, "params") else {},
    }

    # Infer checkpoint path
    checkpoint_path = infer_checkpoint_path(exp_dir, use_best=use_best)

    logger.info(f"  Auto-detected model type: {model_config['type']}")
    logger.info(f"  Auto-detected checkpoint: {checkpoint_path}")
    logger.info(f"  Auto-detected params: {model_config['params']}")

    # Use existing loading function (OLD method)
    return load_pretrained_actor(
        model_config=model_config, checkpoint_path=str(checkpoint_path), device=device, allow_random_init=False, use_weights=use_weights
    )

# ============================================================================
# Original loading functions (backward compatible)
# ============================================================================


def load_pretrained_actor(
    model_config: Union[Dict[str, Any], DictConfig],
    checkpoint_path: str,
    device: Optional[str] = None,
    allow_random_init: bool = False,
    use_weights: bool = True
) -> nn.Module:
    """
    Load a pretrained actor model from checkpoint using the model registry.

    This function:
    1. Instantiates the actor model using the registry (based on model_config.type)
    2. Loads the pretrained weights from checkpoint_path
    3. Moves model to specified device
    4. Sets model to eval mode

    Args:
        model_config: Config dict with 'type' and 'params' keys
                     Example: {"type": "actor_rnt_m", "params": {"num_labels": 1}}
        checkpoint_path: Path to checkpoint file (.pth.tar)
        device: Device to load model on ('cuda' or 'cpu')
        allow_random_init: If True, initialize with random weights if checkpoint missing
                          (default: False - raises error if checkpoint missing)

    Returns:
        Loaded actor model ready for use

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist and allow_random_init=False
        KeyError: If checkpoint doesn't contain 'state_dict' key
        RuntimeError: If state dict doesn't match model architecture

    Example:
        >>> config = {"type": "actor_unet_basic", "params": {"num_labels": 1}}
        >>> actor = load_pretrained_actor(
        ...     model_config=config,
        ...     checkpoint_path="checkpoints/actor.pth.tar",
        ...     device="cuda"
        ... )
        >>> actor.eval()
    """
    # Instantiate model from registry
    # Wrap config in expected format (config.model.type instead of config.type)
    from omegaconf import OmegaConf

    if isinstance(model_config, dict):
        wrapped_config = OmegaConf.create({"model": model_config})
    elif hasattr(model_config, "type"):
        # Already a DictConfig with type at top level, wrap it
        wrapped_config = OmegaConf.create({"model": OmegaConf.to_container(model_config)})
    else:
        # Assume it's already properly structured
        wrapped_config = model_config

    model_type = model_config.get("type") if isinstance(model_config, dict) else model_config.type
    logger.info(f"Creating actor model from registry: {model_type}")
    actor_model = get_actor_from_config(config=wrapped_config)

    if use_weights:
        # Check if checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            if allow_random_init:
                logger.warning(f"⚠️  Checkpoint not found: {checkpoint_path}")
                logger.warning("⚠️  Using RANDOM initialization (allow_random_init=True)")
                logger.warning("⚠️  Model will have random weights - NOT suitable for production!")
                # Model already has random weights from initialization, just continue
            else:
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}\n"
                    f"Please ensure the pretrained actor checkpoint exists at this path.\n"
                    f"Or set allow_random_init=True to use random initialization (testing only)."
                )
        else:
            # Determine device if not explicitly set
            if device is None:
                device = _get_default_device()
            # Load checkpoint
            logger.info(f"Loading actor checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
            # Validate checkpoint format
            if state_dict_key not in checkpoint:
                raise KeyError(
                    f"Checkpoint does not contain 'state_dict' key.\n"
                    f"Found keys: {list(checkpoint.keys())}\n"
                    f"Expected format: {{'state_dict': ..., 'optimizer': ..., ...}}"
                )
            
            # strip param name prefixes from compiled models
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint[state_dict_key].items()}

            # Load state dict
            try:
                actor_model.load_state_dict(state_dict, strict=False)
                logger.info("✓ Actor checkpoint loaded successfully")
            except RuntimeError as e:
                logger.error(f"Failed to load state dict: {e}")
                logger.error(f"Model type: {model_config.get('type')}")
                logger.error(f"Checkpoint path: {checkpoint_path}")
                raise RuntimeError(
                    f"State dict mismatch between model and checkpoint.\n"
                    f"This usually means the checkpoint was saved with a different architecture.\n"
                    f"Model type: {model_config.get('type')}\n"
                    f"Original error: {e}"
                ) from e

    # Move to device and set eval mode
    actor_model = actor_model.to(device)
    actor_model.eval()

    # Count parameters for logging
    num_params = sum(p.numel() for p in actor_model.parameters())
    logger.info(f"✓ Actor model ready ({num_params:,} parameters) on device: {device}")

    return actor_model

def validate_checkpoint_compatibility(model: nn.Module, checkpoint_path: str) -> bool:
    """
    Validate that a checkpoint is compatible with a model architecture.

    This is a dry-run check that loads the checkpoint and attempts to match
    state dict keys without actually loading the weights.

    Args:
        model: Model instance to check compatibility against
        checkpoint_path: Path to checkpoint file

    Returns:
        True if compatible, False otherwise

    Example:
        >>> model = ActorUNetBasic(num_labels=1)
        >>> is_compatible = validate_checkpoint_compatibility(
        ...     model=model,
        ...     checkpoint_path="checkpoints/actor.pth.tar"
        ... )
        >>> if is_compatible:
        ...     print("Checkpoint is compatible!")
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" not in checkpoint:
            logger.warning(f"Checkpoint missing 'state_dict' key: {checkpoint_path}")
            return False

        # Get model and checkpoint keys
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint["state_dict"].keys())

        # Check for missing or extra keys
        missing_keys = model_keys - checkpoint_keys
        extra_keys = checkpoint_keys - model_keys

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            return False

        if extra_keys:
            logger.warning(f"Extra keys in checkpoint: {extra_keys}")
            return False

        logger.info(f"✓ Checkpoint is compatible: {checkpoint_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating checkpoint: {e}")
        return False
