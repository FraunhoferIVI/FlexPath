"""
Adapter for HuggingFace datasets.
"""

from datasets import load_dataset
from typing import Dict, Any, Optional
import logging

from .base_adapter import DataSourceAdapter

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(DataSourceAdapter):
    """Adapter for loading HuggingFace datasets."""

    def __init__(self, repo: str, cache_dir: Optional[str] = None):
        """
        Args:
            repo: HuggingFace repository name
            cache_dir: Optional cache directory
        """
        self.repo = repo
        self.cache_dir = cache_dir
        self._dataset = None
        self._loaded_splits = {}

    def load(self, split: str) -> Dict[str, Any]:
        """Load HuggingFace dataset split."""
        if self._dataset is None:
            logger.info(f"Loading HuggingFace dataset: {self.repo}")
            self._dataset = load_dataset(self.repo, cache_dir=self.cache_dir)

        if split not in self._loaded_splits:
            self._loaded_splits[split] = self._dataset[split]

        return self._loaded_splits[split]

    def get_num_samples(self, split: str) -> int:
        """Return number of samples."""
        data = self.load(split)
        return len(data)

    def get_field(self, split: str, field_name: str, index: int) -> Any:
        """Get a specific field value at index."""
        data = self.load(split)
        return data[field_name][index]
