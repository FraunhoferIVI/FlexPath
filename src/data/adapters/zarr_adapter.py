"""
Adapter for zarr file datasets.
"""

from pathlib import Path
from typing import Dict, Any
import logging
import zipfile

import zarr

from .base_adapter import DataSourceAdapter

logger = logging.getLogger(__name__)


class ZarrAdapter(DataSourceAdapter):
    """Adapter for loading zarr file datasets."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing {split}.zarr files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self._cache = {}

    def _open_zarr_store(self, zarr_path: Path):
        """Open zarr path robustly handling directory and zip stores."""
        # Directory store (typical .zarr directory)
        if zarr_path.is_dir():
            logger.debug("Opening DirectoryStore for %s", zarr_path)
            store = zarr.DirectoryStore(str(zarr_path))
            return zarr.open(store, mode="r")

        # File: check if it's a zip file by magic bytes
        if zarr_path.is_file():
            try:
                with open(zarr_path, "rb") as f:
                    header = f.read(4)
            except Exception as e:
                raise FileNotFoundError(f"Unable to read zarr file header: {zarr_path}") from e

            if header.startswith(b"PK"):
                logger.debug("Detected zip magic bytes, opening ZipStore for %s", zarr_path)
                store = zarr.ZipStore(str(zarr_path), mode="r")
                try:
                    return zarr.open(store, mode="r")
                except zipfile.BadZipFile as e:
                    raise zipfile.BadZipFile(f"Bad zip file for zarr store: {zarr_path}") from e
            else:
                # Not a zip file; attempt generic open and provide clear error if it fails
                logger.debug("File is not a zip. Attempting generic zarr.open for %s", zarr_path)
                try:
                    return zarr.open(str(zarr_path), mode="r")
                except zipfile.BadZipFile as e:
                    # Provide a clearer message
                    raise zipfile.BadZipFile(
                        f"File {zarr_path} is not a valid zarr zip store and could not be opened."
                    ) from e

        # If path doesn't exist or is not accessible
        raise FileNotFoundError(f"zarr path not found or not a file/directory: {zarr_path}")

    def load(self, split: str) -> Dict[str, Any]:
        """Load zarr file for split."""
        if split in self._cache:
            return self._cache[split]

        zarr_path = self.data_dir / f"{split}.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"zarr file not found: {zarr_path}")

        logger.info(f"Loading zarr dataset: {zarr_path}")
        data = self._open_zarr_store(zarr_path)

        # Keep store/group as-is for consistent interface with zarr (lazy arrays)
        self._cache[split] = data

        logger.info("Loaded zarr dataset: %s (keys: %s)", zarr_path, list(data.keys()))
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