"""
Unified model registry for actors and critics.
Extends the existing actor_model_registry.py pattern to support both model types.
"""

from typing import Dict, List, Type
import warnings

import torch.nn as nn
from omegaconf import DictConfig

from src.models.actor.unet_transformer import UnetTransformer


# Actor Registry
ACTOR_REGISTRY: Dict[str, Type[nn.Module]] = {
    "unet_transformer": UnetTransformer,
    "actor_rnt_s_4_transpath_equivalent": UnetTransformer,  # previous name, for backward compatibility
}


def get_actor_from_config(config: DictConfig) -> nn.Module:
    """
    Load actor model from config.

    Args:
        config: Config with model.type and model.params

    Returns:
        Instantiated actor model

    Example:
        config.model.type = "actor_rnt_m"
        config.model.params.in_channels = 3
        model = get_actor_from_config(config)
    """
    model_type = config.model.type

    if model_type not in ACTOR_REGISTRY:
        available = ", ".join(ACTOR_REGISTRY.keys())
        raise ValueError(f"Unknown actor model: '{model_type}'. " f"Available models: {available}")

    model_class = ACTOR_REGISTRY[model_type]
    model_params = dict(config.model.params) if hasattr(config.model, "params") else {}

    try:
        model = model_class(**model_params)
        print(model_params)
        print(f"Successfully loaded actor model: {model_type}")
        return model
    except TypeError as e:
        raise ValueError(
            f"Invalid parameters for {model_type}: {e}. " f"Check config.model.params matches model signature."
        )


def list_available_actors() -> List[str]:
    """Return list of available actor model names."""
    return sorted(ACTOR_REGISTRY.keys())
