"""
Factory for creating datasets from config.
Simplifies dataset instantiation in training scripts.
"""

from omegaconf import DictConfig
import logging

from src.data.adapters import HuggingFaceAdapter, NpzAdapter, ZarrAdapter
from src.data.unified_dataset import ActorDataset

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory for creating datasets from config."""

    @staticmethod
    def create_actor_dataset(config: DictConfig, split: str) -> ActorDataset:
        """
        Create actor dataset from config.

        Args:
            config: Config with data.dataset section
            split: 'train' or 'validation'

        Returns:
            ActorDataset instance

        Example:
            config.data.dataset.source = "hf"
            config.data.dataset.hf_repo = "Cubpaw/voxelgym_5c_42x42_10"
            dataset = DatasetFactory.create_actor_dataset(config, "train")
        """
        # Determine source
        source = config.data.dataset.get("source", "hf")

        # Create adapter
        if source == "hf":
            adapter = HuggingFaceAdapter(
                repo=config.data.dataset.hf_repo,
                cache_dir=config.data.dataset.get("cache_dir", None)
            )
            logger.info(f"Using HuggingFace dataset: {config.data.dataset.hf_repo}")
        elif source == "localnpz":
            adapter = NpzAdapter(data_dir=config.data.dataset.data_dir)
            logger.info(f"Using local NPZ dataset: {config.data.dataset.data_dir}")
        elif source == "localzarr":
            adapter = ZarrAdapter(data_dir=config.data.dataset.data_dir)
            logger.info(f"Using local Zarr dataset: {config.data.dataset.data_dir}")
        else:
            raise ValueError(
                f"Unknown data source: {source}. "
                f"Supported sources: 'hf', 'localnpz'"
            )

        # Determine normalization params
        if hasattr(config.data, 'normalization'):
            if config.data.normalization.get("use_simple_norm", True):
                norm_params = [(0, 0, 0), (1, 1, 1)]
            else:
                norm_params = [
                    tuple(config.data.normalization.mean),
                    tuple(config.data.normalization.std)
                ]
        else:
            norm_params = [(0, 0, 0), (1, 1, 1)]

        # Determine shift probability (only for training)
        shift_p = 0.0
        if split == "train" and hasattr(config.data.dataset, 'shift_p'):
            shift_p = config.data.dataset.shift_p

        # Create dataset
        dataset = ActorDataset(
            source_adapter=adapter,
            split=split,
            image_height=config.data.dataset.image_height,
            image_width=config.data.dataset.image_width,
            norm_params=norm_params,
            shift_p=shift_p,
            disable_augmentation=(split != "train")
        )

        logger.info(f"Created ActorDataset for split '{split}' with {len(dataset)} samples")
        return dataset
