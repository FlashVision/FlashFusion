"""FlashFusion checkpoint utilities — Save and load training checkpoints."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    save_path: Union[str, Path] = "checkpoint.pt",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint.

    Args:
        model: Model to save state_dict from.
        optimizer: Optimizer to save state_dict from (optional).
        epoch: Current epoch number.
        best_metric: Best validation metric achieved so far.
        save_path: File path to save the checkpoint.
        extra: Additional data to store in the checkpoint.

    Returns:
        Path where the checkpoint was saved.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, str(save_path))
    return save_path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load state_dict into (optional).
        optimizer: Optimizer to load state_dict into (optional).
        device: Device to map tensors to.
        strict: Whether to strictly enforce state_dict key matching.

    Returns:
        The full checkpoint dictionary (contains epoch, best_metric, etc.).

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(str(path), map_location=device)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
