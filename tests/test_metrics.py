"""Unit tests for losses and evaluation metrics."""

import math
import numpy as np
import torch

from src.losses import CombinedLoss, SoftDiceLoss
from src.metrics import MetricTracker


def test_losses():
    """Verify CombinedLoss and SoftDiceLoss behavior and gradient flow."""
    criterion = CombinedLoss(num_classes=19, ignore_index=255, ce_weight=1.0, dice_weight=1.0)

    # Dummy logits: (2, 19, 32, 32)
    logits = torch.randn(2, 19, 32, 32, requires_grad=True)

    # Targets containing classes 0..18 and ignore_index 255
    targets = torch.randint(0, 19, (2, 32, 32))
    targets[0, :5, :5] = 255  # Insert void/ignore pixels

    loss = criterion(logits, targets)
    assert not torch.isnan(loss), "Loss returned NaN"
    assert not torch.isinf(loss), "Loss returned Inf"
    assert loss.item() > 0, "Loss must be positive"

    loss.backward()
    assert logits.grad is not None, "Gradients not computed for logits"
    assert not torch.isnan(logits.grad).any(), "NaN found in gradients"
    print("test_losses passed successfully!")


def test_metric_tracker_hand_crafted():
    """Verify MetricTracker with a hand-computed toy confusion matrix."""
    # 3 classes: 0, 1, 2
    tracker = MetricTracker(num_classes=3, ignore_index=255)

    # Construct simple ground truth and predictions
    # Class 0: 4 pixels GT. Pred: 3 correct, 1 predicted as Class 1.
    # Class 1: 3 pixels GT. Pred: 2 correct, 1 predicted as Class 0.
    # Class 2: 0 pixels GT (absent class!).
    # Ignore: 2 pixels with 255.
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 255, 255])
    preds   = np.array([0, 0, 0, 1, 1, 1, 0, 0,   1])

    tracker.update(preds, targets)
    metrics = tracker.compute()

    # Manual calculations:
    # Class 0: TP=3, FP=1, FN=1 -> IoU = 3 / (3 + 1 + 1) = 3/5 = 0.60
    # Class 1: TP=2, FP=1, FN=1 -> IoU = 2 / (2 + 1 + 1) = 2/4 = 0.50
    # Class 2: TP=0, FP=0, FN=0 -> absent, IoU = NaN (excluded from mean)
    # mIoU = (0.60 + 0.50) / 2 = 0.55
    assert abs(metrics["per_class_IoU"]["road"] - 0.60) < 1e-4, f"Expected 0.60, got {metrics['per_class_IoU']['road']}"
    assert abs(metrics["per_class_IoU"]["sidewalk"] - 0.50) < 1e-4, f"Expected 0.50, got {metrics['per_class_IoU']['sidewalk']}"
    assert math.isnan(metrics["per_class_IoU"]["building"]), "Expected NaN for absent class"
    assert abs(metrics["mIoU"] - 0.55) < 1e-4, f"Expected mIoU 0.55, got {metrics['mIoU']}"

    # Total valid pixels evaluated = 7 (ignore pixels excluded)
    assert metrics["total_pixels_evaluated"] == 7
    # Overall pixel accuracy: (3 + 2) / 7 = 5 / 7 = ~0.714285
    assert abs(metrics["pixel_acc"] - 5.0 / 7.0) < 1e-4
    print("test_metric_tracker_hand_crafted passed successfully!")


if __name__ == "__main__":
    test_losses()
    test_metric_tracker_hand_crafted()
    print("All Phase 5 loss and metric tests passed!")
