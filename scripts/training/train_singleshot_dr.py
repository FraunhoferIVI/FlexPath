"""Train DRPG on a single-shot actor environment using the single-shot config.

This script is a Hydra entrypoint (config: ``configs/config_singleshot.yaml``)
and initializes a DRPG training run using a pre-loaded single-shot actor model
wrapped with VSAC logic. It prepares datasets, optimizers and learning-rate
schedule, then runs training via ``DRPG.learn()``.

Key behaviours and outputs
- Loads a pretrained single-shot actor in one of two modes:
    1) Auto-load from a previous experiment (``model.single_shot.actor.experiment_dir``),
    2) Manual load specifying ``type`` + ``checkpoint_path`` in the config.
- Wraps the actor with ``DRPGActorWrapper`` and constructs a ``DRPG`` instance.
- Saves the resolved configuration to ``<exp_dir>/config.yaml`` before training.
- Creates the experiment directory (``cfg.experiment.output_dir``) and writes
    optional logs to ``<exp_dir>/training.log`` when ``cfg.logging.save_to_file`` is true.
- If enabled, tensorboard logs are written to ``<exp_dir>/logs``.
- Checkpoints and training artifacts are written into the chosen ``exp_dir`` (see
    ``cfg.checkpoint.keep_last_n`` to control retention).

How to run
This script uses Hydra for configuration. Typical invocation examples:

    # Use defaults from configs/config_singleshot.yaml
    python scripts/training/train_singleshot_dr.py

    # Override output directory and total epochs via Hydra CLI
    python scripts/training/train_singleshot_dr.py experiment.output_dir=experiments/runs/myrun training.hyperparams.total_epochs=100

Notes
- Random seeds are taken from ``cfg.experiment.seed`` and applied to Python, NumPy
    and PyTorch for reproducibility.
- Actor loading supports an auto-load format (recommended) and a manual format
    for backward compatibility; misconfiguration will raise a ValueError with
    guidance on required keys.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.models.policies.CustomDRPG import DRPG

from src.models.policies.drpg.actor_wrapper import DRPGActorWrapper
from src.data.dataset_factory import DatasetFactory
from src.utils.lr_schedules import make_delayed_cosine_schedule


from src.utils.checkpoint_loader import (
    load_pretrained_actor,
    load_pretrained_actor_from_experiment,
)

import random

import torch


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


@hydra.main(version_base=None, config_path="../../configs", config_name="config_singleshot")
def main(cfg: DictConfig):
    """Main training function."""

    # Create experiment directory first
    exp_dir = Path(cfg.experiment.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler to root logger (Hydra set up console already)
    log_file = exp_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    # Get logger
    logger = logging.getLogger(__name__)

    # Print configuration
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logger.info("=" * 80)

    # Seed everything
    # python and numpy
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)

    # torch
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.experiment.seed)
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Enable TF32 globally
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    # =========================================================================
    # Actor Model Loading
    # =========================================================================
    # Two loading modes supported:
    #   1. Auto-loading (recommended): Specify experiment_dir + use_best
    #      - Auto-loads model type and params from experiment config
    #      - Simpler, less error-prone
    #   2. Manual (backward compatible): Specify type + checkpoint_path + params
    #      - Manual configuration
    #      - Kept for backward compatibility
    # =========================================================================

    logger.info("=" * 80)
    logger.info("Loading pretrained actor model...")

    # Check which loading mode to use
    if hasattr(cfg.model.single_shot.actor, 'experiment_dir'):
        # Auto-load from experiment directory
        logger.info("Using auto-load format (experiment_dir)")
        logger.info(f"  Experiment dir: {cfg.model.single_shot.actor.experiment_dir}")
        use_best = cfg.model.single_shot.actor.get('use_best', True)
        logger.info(f"  Use best checkpoint: {use_best}")

        actor_model = load_pretrained_actor_from_experiment(
            exp_dir=cfg.model.single_shot.actor.experiment_dir,
            use_best=use_best,
            device=cfg.experiment.device,
            use_weights=cfg.model.single_shot.actor.get('use_weights', True)
        )
        # Extract model type and resolution from experiment config
        from src.utils.checkpoint_loader import load_config_from_experiment_dir
        actor_exp_config = load_config_from_experiment_dir(cfg.model.single_shot.actor.experiment_dir)
        actor_model_type = actor_exp_config.model.type
    elif hasattr(cfg.model.single_shot.actor, 'type'):
        # Manual loading: Explicit type + checkpoint_path + params
        logger.info("Using manual format (explicit type + checkpoint_path + params)")
        logger.info(f"  Actor type: {cfg.model.single_shot.actor.type}")
        logger.info(f"  Checkpoint: {cfg.model.single_shot.actor.checkpoint_path}")

        actor_model = load_pretrained_actor(
            model_config=cfg.model.single_shot.actor,
            checkpoint_path=cfg.model.single_shot.actor.checkpoint_path,
            device=cfg.experiment.device,
            use_weights=True
        )
        actor_model_type = cfg.model.single_shot.actor.type
    else:
        raise ValueError(
            "Invalid actor config format. Must have either:\n"
            "  - 'experiment_dir' (auto-load format, recommended), OR\n"
            "  - 'type' + 'checkpoint_path' + 'params' (manual format, backward compatible)\n"
            f"Found keys: {list(cfg.model.single_shot.actor.keys())}"
        )

    actor_wrapper = DRPGActorWrapper(
        actor_model=actor_model,
        exploration_strategy=cfg.training.exploration.strategy
    )

    # set exploration strategy hyperparams
    actor_wrapper.modify_exploration_strategy_params(
        **cfg.training.exploration.hyperparams
    )

    logger.info("✓ Actor ready for DRPG")
    logger.info("=" * 80)

    # Save resolved config with policy_kwargs BEFORE creating environment
    config_save_path = exp_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved configuration to {config_save_path}")

    # Create datasets
    train_dataset = DatasetFactory.create_actor_dataset(cfg, split="train")
    val_dataset = DatasetFactory.create_actor_dataset(cfg, split="validation")

    # Create model
    logger.info("Initializing DRPG model (CustomDRPG)...")
    logger.info(f"  Actor: Pre-loaded {actor_model_type}")

    actor_optimizer = torch.optim.AdamW(
        params=actor_wrapper.actor_model.parameters(),
        lr=cfg.training.hyperparams.learning_rate,
        fused=True
    )

    # init lr schedule, warmup -> linear -> cosine decay
    lr_schedule = make_delayed_cosine_schedule(
        optimizer=actor_optimizer,
        total_epochs=cfg.training.hyperparams.total_epochs,
        warmup_epochs=cfg.training.hyperparams.warmup_epochs,
        cosine_decay_start_epoch=cfg.training.hyperparams.cosine_decay_start,
        end_lr=cfg.training.hyperparams.end_lr
    )

    model = DRPG(
        exp_dir=exp_dir,
        exp_name=cfg.experiment.name,
        reward_func=cfg.training.hyperparams.reward_f,
        policy=actor_wrapper,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizer=actor_optimizer,
        lr_schedule=lr_schedule,
        batch_size=cfg.training.hyperparams.batch_size,
        eval_freq=cfg.training.hyperparams.batches_per_epoch,
        total_epochs=cfg.training.hyperparams.total_epochs,
        keep_last_n_checkpoints=cfg.checkpoint.keep_last_n,
        adaptive_pixel_penalty_scaling=cfg.training.hyperparams.adaptive_pixel_penalty_scaling,
        pixel_penalty_scaling_start_step=cfg.training.hyperparams.pixel_penalty_scaling_start_step,
        pixel_penalty_scaling_start_val=cfg.training.hyperparams.pixel_penalty_scaling_start_val,
        pixel_penalty_scaling_stop_val=cfg.training.hyperparams.pixel_penalty_scaling_stop_val,
        obstacle_penalty_scaling=cfg.training.hyperparams.obstacle_penalty_scaling,
        tensorboard_log=str(exp_dir / "logs"),
        device=cfg.experiment.device,
        use_mixed_precision=cfg.training.hyperparams.use_mixed_precision,
        precision=cfg.training.hyperparams.precision,
    )

    # Training
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info("=" * 80)

    model.learn()

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
