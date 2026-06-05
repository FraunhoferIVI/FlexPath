"""
Unified model registry for actors and critics.
Extends the existing actor_model_registry.py pattern to support both model types.
"""

from typing import Dict, List, Type
import warnings

import torch.nn as nn
from omegaconf import DictConfig

from src.models.actor.actor_rnt_s_4_transpath_equivalent import ActorRNT_S_4_TP_Equiv

from src.models.actor.iastar.iastar import iastar
from src.models.actor.daastar.training import DAAStarPlannerModule

from src.models.actor.neuralastar.astar import NeuralAstar

from src.models.actor.transPath.autoencoder import Autoencoder


# Actor Registry
ACTOR_REGISTRY: Dict[str, Type[nn.Module]] = {
    # New names (preferred)
    "transpath_autoencoder": Autoencoder,
    "actor_rnt_s_4_transpath_equivalent": ActorRNT_S_4_TP_Equiv,
    # Backward compatibility: Old class names (deprecated in the next PR)
    "iastar": iastar,
    "neuralastar": NeuralAstar,
    "daastar": DAAStarPlannerModule
}

# Deprecated actor model names (for backward compatibility)
DEPRECATED_ACTOR_NAMES = {
    "PathFindingNetworkUNET2": "actor_unet_basic",
    "ActorUnetAtt": "actor_rnt_m",
    "ActorUnetAtt_s": "actor_rnt_s",
    "ActorUnetAtt_L": "actor_rnt_l",
    "ActorUnetAtt_T3": "actor_rnt_t3",
    "ActorUnetCrossAttFuse": "actor_rn_cross_att",
}

# Deprecated critic model names (for backward compatibility)
DEPRECATED_CRITIC_NAMES = {
    "PathFindingNetworkCritic": "critic_concat",
    "PathFindingNetworkCritic1": "critic_separate",
    "PathFindingNetworkCritic2": "critic_separate_1x1",
    "PathFindingNetworkCritic3": "critic_double_conv",
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

    # Warn if using deprecated name
    if model_type in DEPRECATED_ACTOR_NAMES:
        new_name = DEPRECATED_ACTOR_NAMES[model_type]
        warnings.warn(
            f"Actor model name '{model_type}' is deprecated and will be removed in a future release. "
            f"Please use '{new_name}' instead. "
            f"Update your config: model.type = '{new_name}'",
            DeprecationWarning,
            stacklevel=2
        )

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
