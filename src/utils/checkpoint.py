"""Checkpoint management utilities."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import torch


class CheckpointManager:
    """Manages checkpoint saving and loading with metadata."""

    def __init__(self, exp_dir: str, keep_last_n: int = 5):
        """
        Initialize checkpoint manager.

        Args:
            exp_dir: Experiment directory path
            keep_last_n: Number of checkpoints to keep (excluding best)
        """
        self.exp_dir = Path(exp_dir)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

        self.best_metric = float("-inf")
        self.checkpoints = []  # List of (step, path) tuples
    
    def save_checkpoint(
        self,
        model,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        metric_value: Optional[float] = None,
    ):
        """
        Save a checkpoint with metadata.

        Args:
            model: Model or dict of state dicts
            step: Training step
            metadata: Additional metadata to save
            is_best: Whether this is the best checkpoint
            metric_value: Metric value for tracking best model
        """
        # Update best metric
        if metric_value is not None and metric_value > self.best_metric:
            self.best_metric = metric_value
            is_best = True

        # Prepare checkpoint data
        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Save model (handle both single model and dict)
        if hasattr(model, "save"):
            # StableBaselines3 model
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.zip"
            model.save(str(checkpoint_path))
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pth"
            if isinstance(model, dict):
                checkpoint_data.update(model)
            else:
                checkpoint_data["model_state_dict"] = model.state_dict()
            torch.save(checkpoint_data, checkpoint_path)

        # Save metadata separately
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Track checkpoint
        self.checkpoints.append((step, checkpoint_path))

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"best{checkpoint_path.suffix}"
            shutil.copy2(checkpoint_path, best_path)
            shutil.copy2(metadata_path, best_path.with_suffix(".json"))

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"latest{checkpoint_path.suffix}"
        shutil.copy2(checkpoint_path, latest_path)
        shutil.copy2(metadata_path, latest_path.with_suffix(".json"))

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self.checkpoints) > self.keep_last_n:
            # Sort by step
            self.checkpoints.sort(key=lambda x: x[0])

            # Remove oldest checkpoints
            # Special case: if keep_last_n=0, remove all numbered checkpoints
            if self.keep_last_n == 0:
                to_remove = self.checkpoints[:]
            else:
                to_remove = self.checkpoints[: -self.keep_last_n]

            for step, path in to_remove:
                if path.exists():
                    path.unlink()
                metadata_path = path.with_suffix(".json")
                if metadata_path.exists():
                    metadata_path.unlink()

            # Keep only recent checkpoints in list
            if self.keep_last_n == 0:
                self.checkpoints = []
            else:
                self.checkpoints = self.checkpoints[-self.keep_last_n :]

    def load_checkpoint(self, checkpoint_name: str = "latest"):
        """
        Load a checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to load ('latest', 'best', or specific step)

        Returns:
            Tuple of (checkpoint_data, metadata)
        """
        if checkpoint_name in ["latest", "best"]:
            # Try .zip first (SB3 format), then .pth (PyTorch format)
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.zip"
            if not checkpoint_path.exists():
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        else:
            checkpoint_path = Path(checkpoint_name)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load metadata
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        # Load checkpoint based on format
        if checkpoint_path.suffix == ".zip":
            # SB3 format - return path for model.load()
            return str(checkpoint_path), metadata
        else:
            # PyTorch format
            checkpoint_data = torch.load(checkpoint_path)
            return checkpoint_data, metadata

    def get_latest_step(self) -> Optional[int]:
        """Get the step number of the latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest.json"
        if latest_path.exists():
            with open(latest_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("step")
        return None


def save_pretrained_weights(model: torch.nn.Module, save_path: str, metadata: Optional[Dict] = None):
    """
    Save pretrained model weights with metadata.

    Args:
        model: PyTorch model
        save_path: Path to save weights
        metadata: Optional metadata dict
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_path)
    print(f"Saved pretrained weights to {save_path}")


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: str, strict: bool = True):
    """
    Load pretrained weights into a model.

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Metadata dict if available
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        metadata = checkpoint.get("metadata", {})
    else:
        model.load_state_dict(checkpoint, strict=strict)
        metadata = {}

    print(f"Loaded pretrained weights from {checkpoint_path}")
    return metadata
