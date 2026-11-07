"""Multi-class loss functions for nighttime semantic segmentation.

Combines Cross-Entropy (with ignore_index=255) and multi-class Soft-Dice loss.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class Soft-Dice Loss with ignore_index support.

    Computes soft dice over softmax probabilities across all non-ignored classes.
    """

    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 255,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Soft-Dice loss.

        Args:
            logits: Predicted raw logits of shape (B, num_classes, H, W).
            targets: Ground truth integer labels of shape (B, H, W).

        Returns:
            Scalar loss tensor.
        """
        # (B, C, H, W) -> probabilities via softmax
        probs = F.softmax(logits, dim=1)

        # Valid mask: exclude ignore_index (255)
        valid_mask = targets != self.ignore_index  # (B, H, W)

        # Clamp targets for one-hot encoding (ignore pixels mapped temporarily to 0)
        clamped_targets = targets.clone()
        clamped_targets[~valid_mask] = 0

        # One-hot encode: (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
        one_hot = (
            F.one_hot(clamped_targets, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Zero out ignored regions in both probs and one_hot
        mask_expanded = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * mask_expanded
        one_hot = one_hot * mask_expanded

        # Compute intersection and union per class over spatial dimensions (and batch)
        intersection = torch.sum(probs * one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + one_hot, dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Only average classes present in the valid one_hot ground truth
        present_classes = torch.sum(one_hot, dim=(0, 2, 3)) > 0
        if present_classes.sum() > 0:
            loss = 1.0 - dice_per_class[present_classes].mean()
        else:
            loss = 1.0 - dice_per_class.mean()

        return loss


class CombinedLoss(nn.Module):
    """Weighted sum of Cross-Entropy and Soft-Dice Loss.

    L_total = ce_weight * L_ce + dice_weight * L_dice
    """

    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        self.dice_loss = SoftDiceLoss(
            num_classes=num_classes, ignore_index=ignore_index
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: (B, num_classes, H, W)
            targets: (B, H, W)

        Returns:
            Scalar loss tensor.
        """
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        total_loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(
                f"Loss is NaN or Inf! CE: {loss_ce.item()}, Dice: {loss_dice.item()}"
            )

        return total_loss
