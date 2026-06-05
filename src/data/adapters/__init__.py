"""
Data source adapters for unified dataset loading.
"""

from .base_adapter import DataSourceAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .npz_adapter import NpzAdapter
from .zarr_adapter import ZarrAdapter

__all__ = [
    "DataSourceAdapter",
    "HuggingFaceAdapter",
    "NpzAdapter",
    "ZarrAdapter",
]
