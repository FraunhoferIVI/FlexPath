"""
Abstract base class for dataset adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class DataSourceAdapter(ABC):
    """
    Abstract adapter for different data sources.
    Implements adapter pattern to unify HuggingFace and NPZ loading.
    """

    @abstractmethod
    def load(self, split: str) -> Dict[str, Any]:
        """
        Load dataset for given split.

        Args:
            split: Dataset split ("train", "validation", "test")

        Returns:
            Dictionary with dataset fields (structure depends on implementation)
        """
        pass

    @abstractmethod
    def get_num_samples(self, split: str) -> int:
        """Return number of samples in split."""
        pass

    @abstractmethod
    def get_field(self, split: str, field_name: str, index: int) -> Any:
        """
        Get a specific field value at index.

        Args:
            split: Dataset split
            field_name: Name of the field (e.g., "image", "path_label")
            index: Sample index

        Returns:
            Field value at the given index
        """
        pass
