"""Checkpoint saving, loading, and resume helpers."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    out_dir: str,
    filename: str = "last.pth",
) -> None:
    """Save training state checkpoint.

    Args:
        state: Dict containing epoch, model, optimizer, scheduler, rng_state, best_metric.
        is_best: Whether this checkpoint achieved the best validation score.
        out_dir: Directory to save checkpoints.
        filename: Checkpoint filename (default last.pth).
    """
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_path = ckpt_dir / filename
    torch.save(state, last_path)

    if is_best:
        best_path = ckpt_dir / "best.pth"
        torch.save(state, best_path)


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint for inference or resuming training.

    Args:
        ckpt_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Load model weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip 'module.' prefix if saved with DataParallel
    cleaned_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
