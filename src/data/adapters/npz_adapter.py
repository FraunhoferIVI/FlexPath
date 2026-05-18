"""
Adapter for NPZ file datasets.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

from .base_adapter import DataSourceAdapter

logger = logging.getLogger(__name__)


class NpzAdapter(DataSourceAdapter):
    """Adapter for loading NPZ file datasets."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing {split}.npz files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self._cache = {}

    def load(self, split: str) -> Dict[str, Any]:
        """Load NPZ file for split."""
        if split in self._cache:
            return self._cache[split]

        npz_path = self.data_dir / f"{split}.npz"  # change back
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        logger.info(f"Loading NPZ dataset: {npz_path}")
        data = np.load(npz_path, allow_pickle=False) #, mmap_mode='r'

        # Convert to dict for consistent interface
        self._cache[split] = data  # {key: data[key] for key in data.files}

        return self._cache[split]

    def get_num_samples(self, split: str) -> int:
        """Return number of samples."""
        data = self.load(split)
        # Assume first array contains all samples
        first_key = list(data.keys())[0]
        return len(data[first_key])

    def get_field(self, split: str, field_name: str, index: int) -> Any:
        """Get a specific field value at index."""
        data = self.load(split)
        return data[field_name][index]
