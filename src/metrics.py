"""Evaluation metrics for semantic segmentation.

Accumulates dataset-level confusion matrix to compute mIoU, per-class IoU,
Dice, mean pixel accuracy (mPA), overall accuracy, sensitivity, and specificity.
Includes NaN-safe handling for absent classes (e.g. bus).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

# 19 Cityscapes / ACDC classes
CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


class MetricTracker:
    """Accumulates 2D confusion matrix to compute exact dataset-level segmentation metrics."""

    def __init__(self, num_classes: int = 19, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Matrix shape: (num_classes, num_classes) where mat[gt, pred] = count
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset the accumulated confusion matrix."""
        self.mat.fill(0)

    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Update confusion matrix with a batch of predictions and ground truths.

        Args:
            preds: (B, H, W) or (H, W) predicted class labels in [0..num_classes-1].
            targets: (B, H, W) or (H, W) ground truth labels in [0..num_classes-1, ignore_index].
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = preds.flatten()
        targets = targets.flatten()

        # Filter out ignored pixels (255)
        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        t_valid = targets[valid_mask]
        p_valid = preds[valid_mask]

        # Filter any out-of-range predictions
        pred_valid = (p_valid >= 0) & (p_valid < self.num_classes)
        t_valid = t_valid[pred_valid]
        p_valid = p_valid[pred_valid]

        # Accumulate: index = target * num_classes + pred
        indices = t_valid * self.num_classes + p_valid
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.mat += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute all metrics from accumulated confusion matrix.

        Returns:
            Dictionary matching the common results schema:
            mIoU, per_class_IoU, dice_mean, per_class_dice, mPA,
            pixel_acc, sensitivity_macro, specificity_macro.
        """
        mat = self.mat.astype(np.float64)
        total_pixels = np.sum(mat)

        tp = np.diag(mat)  # (C,)
        fp = np.sum(mat, axis=0) - tp  # (C,)
        fn = np.sum(mat, axis=1) - tp  # (C,)
        tn = total_pixels - tp - fp - fn  # (C,)

        # Per-class IoU: TP / (TP + FP + FN)
        union = tp + fp + fn
        per_class_iou = {}
        valid_ious = []

        # Per-class Dice: 2*TP / (2*TP + FP + FN)
        per_class_dice = {}
        valid_dices = []

        # Sensitivity (Recall): TP / (TP + FN)
        per_class_sens = {}
        valid_sens = []

        # Specificity: TN / (TN + FP)
        per_class_spec = {}
        valid_spec = []

        for c in range(self.num_classes):
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"

            # IoU
            if union[c] > 0:
                iou_c = float(tp[c] / union[c])
                per_class_iou[name] = iou_c
                valid_ious.append(iou_c)
            else:
                # Class absent in both GT and prediction
                per_class_iou[name] = float("nan")

            # Dice
            dice_denom = 2.0 * tp[c] + fp[c] + fn[c]
            if dice_denom > 0:
                dice_c = float(2.0 * tp[c] / dice_denom)
                per_class_dice[name] = dice_c
                valid_dices.append(dice_c)
            else:
                per_class_dice[name] = float("nan")

            # Sensitivity
            gt_c = tp[c] + fn[c]
            if gt_c > 0:
                sens_c = float(tp[c] / gt_c)
                per_class_sens[name] = sens_c
                valid_sens.append(sens_c)
            else:
                per_class_sens[name] = float("nan")

            # Specificity
            non_gt_c = tn[c] + fp[c]
            if non_gt_c > 0:
                spec_c = float(tn[c] / non_gt_c)
                per_class_spec[name] = spec_c
                valid_spec.append(spec_c)
            else:
                per_class_spec[name] = float("nan")

        # Macro averages excluding NaN (absent classes)
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        dice_mean = float(np.mean(valid_dices)) if valid_dices else 0.0
        mpa = float(np.mean(valid_sens)) if valid_sens else 0.0
        sensitivity_macro = mpa
        specificity_macro = float(np.mean(valid_spec)) if valid_spec else 0.0
        pixel_acc = float(np.sum(tp) / total_pixels) if total_pixels > 0 else 0.0

        return {
            "mIoU": miou,
            "per_class_IoU": per_class_iou,
            "dice_mean": dice_mean,
            "per_class_dice": per_class_dice,
            "mPA": mpa,
            "pixel_acc": pixel_acc,
            "sensitivity_macro": sensitivity_macro,
            "specificity_macro": specificity_macro,
            "total_pixels_evaluated": int(total_pixels),
        }
