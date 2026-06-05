"""
Entry point script for actor pretraining.
Uses Hydra for configuration management.

Usage:
    python scripts/pretraining/train_actor.py
    python scripts/pretraining/train_actor.py training.batch_size=64
    python scripts/pretraining/train_actor.py training.num_epochs=20 training.learning_rate=1e-4
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging

import hydra
from omegaconf import DictConfig

from src.data.dataset_factory import DatasetFactory
from src.training.train_actor_loop import train_actor
from src.utils.common import count_parameters, seed_everything
from src.utils.model_registry import get_actor_from_config
from src.utils.script_utils import attach_file_logger, ensure_experiment_dir, log_config, resolve_device


@hydra.main(config_path="../../configs/pretraining", config_name="config_actor", version_base=None)
def main(config: DictConfig):
    """Main training function with Hydra config."""

    # Create experiment directory and file logger first
    exp_dir = ensure_experiment_dir(config.experiment.output_dir)
    attach_file_logger(config, exp_dir)

    # Get logger
    logger = logging.getLogger(__name__)
    log_config(logger, config, title="Actor (UNET) Pretraining")

    # Set seed
    seed_everything(config.experiment.seed)

    # Set device
    device = resolve_device(config.experiment.device, logger)

    # =========================================================================
    # Dataset Loading with Factory Pattern
    # =========================================================================
    # The DatasetFactory automatically handles:
    # - Both HuggingFace and NPZ data sources (based on config.data.dataset.source)
    # - Normalization params (from config.data.normalization)
    # - Image dimensions (from config.data.dataset.image_height/width)
    # - Augmentation (automatically disabled for validation)
    #
    # To switch data sources, just change config.data.dataset.source:
    #   - "hf" for HuggingFace (requires config.data.dataset.hf_repo)
    #   - "localnpz" for local NPZ files (requires config.data.dataset.data_dir)
    # =========================================================================

    logger.info("Creating datasets using DatasetFactory...")
    train_dataset = DatasetFactory.create_actor_dataset(config, split="train")
    val_dataset = DatasetFactory.create_actor_dataset(config, split="validation")
    logger.info("✓ Datasets created successfully")

    # Check diagonal movements config (if needed for metrics)
    diagonal_movements_at_obstacle = False
    if hasattr(config.data.dataset, "diagonal_movements_at_obstacle"):
        if config.data.dataset.diagonal_movements_at_obstacle:
            diagonal_movements_at_obstacle = True

    # Create model using registry
    logger.info(f"Creating model: {config.model.type}")
    model = get_actor_from_config(config=config)

    # Count parameters
    num_params = count_parameters(model)
    logger.info(f"✓ Model created ({num_params:,} trainable parameters)\n")

    # Train
    trained_model = train_actor(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        diagonal_movements_at_obstacle=diagonal_movements_at_obstacle,
        compile_model=config.training.get("compile_model", True)
    )

    logger.info("✓ Script completed successfully!")


if __name__ == "__main__":
    main()
