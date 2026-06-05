"""
Unified dataset implementation using adapter pattern.
Replaces actor_dataset.py, actor_npz_dataset.py, and critic_dataset.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
import logging

from src.data.adapters.base_adapter import DataSourceAdapter
from src.utils.transform_utils import (
    get_unet_train_transform,
    get_unet_val_transform,
    shift_image,
)


logger = logging.getLogger(__name__)


class VoxelGymDataset(Dataset):
    """
    Base unified dataset class using adapter pattern.
    Supports multiple data sources (HuggingFace, NPZ) through adapters.
    """

    def __init__(
        self,
        source_adapter: DataSourceAdapter,
        split: str,
        image_height: int,
        image_width: int,
        norm_params: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
        shift_p: float = 0.0,
        disable_augmentation: bool = False
    ):
        """
        Args:
            source_adapter: Data source adapter (HF or NPZ)
            split: Dataset split ('train', 'validation', 'test')
            image_height: Target image height
            image_width: Target image width
            norm_params: (mean, std) for normalization. If None, uses (0,0,0), (1,1,1)
            shift_p: Probability of shift augmentation (only for training)
            disable_augmentation: Force disable augmentation
        """
        self.source_adapter = source_adapter
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.shift_p = shift_p if split == "train" and not disable_augmentation else 0.0
        self.disable_augmentation = disable_augmentation

        # Set norm params
        if norm_params is None:
            norm_params = [(0, 0, 0), (1, 1, 1)]  # Simple normalization
        self.norm_params = norm_params

        # Load data through adapter
        self.data = source_adapter.load(split)

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self)} samples "
            f"(split={split}, augmentation={'enabled' if not disable_augmentation and split == 'train' else 'disabled'})"
        )

    def __len__(self) -> int:
        """Return number of samples in the current split."""
        return self.source_adapter.get_num_samples(self.split)

    def _extract_start_end(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract start and end coordinates from a normalized RGB image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Normalized image tensor of shape ``[3, H, W]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Start and end indices, each with shape ``[2]``.

        Raises
        ------
        ValueError
            If start or end markers are missing in the image.
        """
        start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255])
        end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255])

        # Find start and end pixels
        find_start = torch.all(torch.isclose(image_tensor, start_color.view(-1, 1, 1), atol=1.0 / 255.0), dim=0)
        find_end = torch.all(torch.isclose(image_tensor, end_color.view(-1, 1, 1), atol=1.0 / 255.0), dim=0)

        start_indices = torch.nonzero(find_start, as_tuple=False).squeeze()
        end_indices = torch.nonzero(find_end, as_tuple=False).squeeze()

        # Handle edge cases
        if start_indices.dim() > 1:
            if start_indices.shape[0] == 0:
                logger.warning("Sample has no start position: Sample skipped!")
                raise ValueError("No start position found")
            else:
                start_indices = start_indices[0]

        if end_indices.dim() > 1:
            if end_indices.shape[0] == 0:
                logger.warning("Sample has no end position: Sample skipped!")
                raise ValueError("No end position found")
            else:
                end_indices = end_indices[0]

        return start_indices, end_indices

    def __getitem__(self, idx: int):
        """Must be implemented by subclasses."""
        raise NotImplementedError


class ActorDataset(VoxelGymDataset):
    """
    Unified dataset for actor (policy) training.
    Supports both HuggingFace and NPZ data sources.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the actor dataset.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :class:`VoxelGymDataset`.
        """
        super().__init__(*args, **kwargs)

        # Set transforms
        if self.split == "train" and not self.disable_augmentation:
            self.transform = get_unet_train_transform(
                self.image_height, self.image_width, self.norm_params
            )

        else:
            self.transform = get_unet_val_transform(
                self.image_height, self.image_width, self.norm_params
            )


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch a sample and extract start/end coordinates.

        Samples without valid start/end markers are skipped by advancing to the
        next index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict
            Keys:
            - ``images``: torch.Tensor [3, H, W]
            - ``labels``: torch.Tensor [H, W]
            - ``starts``: torch.Tensor [2]
            - ``ends``: torch.Tensor [2]
            - ``unnormalized_image``: torch.Tensor [H, W, 3]

        Raises
        ------
        StopIteration
            If no valid sample is found after the final index.
        """
        try:
            # Load from adapter
            image = np.asarray(self.source_adapter.get_field(self.split, "image", idx))
            label = np.asarray(self.source_adapter.get_field(self.split, "path_label", idx))

            # Convert label to uint8 if needed (for NPZ compatibility)
            if label.dtype == bool:
                label = label.astype(np.uint8)

            # Apply shift augmentation if enabled
            if self.shift_p > 0:
                image, label = shift_image(image, label, p=self.shift_p)

            # Apply transforms
            transformed = self.transform(image=image, mask=label)
            image_tensor = transformed["image"]
            label_tensor = transformed["mask"]

            # Extract start/end
            start_indices, end_indices = self._extract_start_end(image_tensor)

            return {
                "images": image_tensor,
                "labels": label_tensor,
                "starts": start_indices,
                "ends": end_indices,
                "unnormalized_image": torch.from_numpy(image)
            }

        except ValueError:
            # No start/end found - skip to next sample
            if (idx + 1) < len(self):
                return self.__getitem__(idx + 1)
            else:
                raise StopIteration
