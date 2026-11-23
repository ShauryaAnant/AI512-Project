"""Training and evaluation engine loops."""

import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src.metrics import MetricTracker


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    clip_grad: Optional[float] = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    logger: Optional[logging.Logger] = None,
    teacher_model: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """Train for a single epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        criterion: Loss function (standard CombinedLoss or KnowledgeDistillationLoss).
        optimizer: Optimizer.
        device: Torch device.
        epoch: Current epoch index.
        clip_grad: Max grad norm for clipping.
        scaler: Optional AMP GradScaler.
        logger: Optional logger.
        teacher_model: Optional frozen teacher model for Knowledge Distillation.

    Returns:
        (average_epoch_loss, average_grad_norm)
    """
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(
        dataloader,
        desc=f"Train Ep [{epoch + 1:03d}]",
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar):
        images, targets, _ = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Compute teacher logits if Knowledge Distillation is active
        teacher_logits = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(images)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                if teacher_logits is not None:
                    loss = criterion(logits, targets, teacher_logits)
                else:
                    loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                total_grad_norm += float(grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if teacher_logits is not None:
                loss = criterion(logits, targets, teacher_logits)
            else:
                loss = criterion(logits, targets)

            loss.backward()
            if clip_grad is not None:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                total_grad_norm += float(grad_norm)
            optimizer.step()

        loss_val = loss.item()
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Divergence detected: loss is {loss_val} at epoch {epoch}, step {step}")

        total_loss += loss_val
        running_avg = total_loss / (step + 1)
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{running_avg:.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    avg_norm = total_grad_norm / max(num_batches, 1)
    return avg_loss, avg_norm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    eval_orig_res: bool = True,
) -> Tuple[Dict[str, Any], float]:
    """Evaluate model over dataloader using accumulated confusion matrix.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation dataloader.
        criterion: Optional loss function.
        device: Torch device.
        eval_orig_res: If True, upsamples logits to target resolution before argmax.

    Returns:
        (metrics_dict, average_val_loss)
    """
    model.eval()
    tracker = MetricTracker(num_classes=19, ignore_index=255)
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)

    for batch in pbar:
        images, targets, _ = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)

        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item()

        # If evaluating at original target resolution
        if eval_orig_res and logits.shape[-2:] != targets.shape[-2:]:
            logits = nn.functional.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        preds = torch.argmax(logits, dim=1)
        tracker.update(preds, targets)

    metrics = tracker.compute()
    avg_loss = total_loss / max(num_batches, 1) if criterion is not None else 0.0
    return metrics, avg_loss
