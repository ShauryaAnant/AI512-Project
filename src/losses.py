"""Multi-class loss functions for nighttime semantic segmentation.

Includes:
1. SoftDiceLoss: Multi-class Soft-Dice with ignore_index support.
2. CombinedLoss: Baseline weighted sum of CE + Soft-Dice (retained bit-identical).
3. OHEMCrossEntropyLoss: Online Hard Example Mining with class-frequency rebalancing.
4. ClassWeightedBoundaryLoss: Mentor-proposed boundary loss with small-object focus.
5. BOLTLoss: Unified novel training objective combining OHEM, Soft-Dice, and Boundary Loss.
6. build_loss(cfg): Factory function selecting baseline or BOLT loss based on config.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ACDC Night train split pixel distribution (from Phase 2 audit of 400 images)
ACDC_NIGHT_TRAIN_PIXELS = [
    251770125,  # 0: road (30.35%)
    61088259,   # 1: sidewalk (7.37%)
    147157443,  # 2: building (17.74%)
    14054749,   # 3: wall (1.69%)
    10223985,   # 4: fence (1.23%)
    7845730,    # 5: pole (0.95%)
    1751850,    # 6: traffic light (0.21%)
    3100472,    # 7: traffic sign (0.37%)
    94131506,   # 8: vegetation (11.35%)
    5435223,    # 9: terrain (0.66%)
    119175815,  # 10: sky (14.37%)
    1227313,    # 11: person (0.15%)
    325000,     # 12: rider (0.04%)
    17716814,   # 13: car (2.14%)
    177153,     # 14: truck (0.02%)
    1746314,    # 15: bus (0.21%)
    5993415,    # 16: train (0.72%)
    345828,     # 17: motorcycle (0.04%)
    651240,     # 18: bicycle (0.08%)
]


def get_rcs_mild_class_weights(json_path: str = "data/processed/rcs_training_stats.json") -> torch.Tensor:
    """Load or compute mild inverse-square-root class weights: w_c ~ 1 / sqrt(f_c) clipped at 5.0."""
    p = Path(json_path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / json_path
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            weights = np.array(data["mild_class_weights"], dtype=np.float32)
            return torch.from_numpy(weights)

    fallback = [
        0.1970, 0.1970, 0.1970, 0.3983, 0.4670, 0.5331, 1.1282, 0.8481,
        0.1970, 0.6405, 0.1970, 1.3479, 2.6194, 0.3548, 3.5479, 1.1300,
        0.6100, 2.5393, 1.8504
    ]
    return torch.tensor(fallback, dtype=torch.float32)


def get_acdc_class_weights(c_smooth: float = 1.02, num_classes: int = 19) -> torch.Tensor:
    """Compute normalized smoothed inverse-frequency class weights for ACDC night.

    Formula: w_c = 1.0 / ln(c_smooth + p_c), normalized so mean(w) = 1.0.
    """
    counts = np.array(ACDC_NIGHT_TRAIN_PIXELS[:num_classes], dtype=np.float64)
    total = np.sum(counts)
    p = counts / max(total, 1.0)
    w = 1.0 / np.log(c_smooth + p)
    w_norm = num_classes * (w / np.sum(w))
    return torch.from_numpy(w_norm).float()


class SoftDiceLoss(nn.Module):
    """Multi-class Soft-Dice Loss with ignore_index support."""

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
        probs = F.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index

        clamped_targets = targets.clone()
        clamped_targets[~valid_mask] = 0

        one_hot = (
            F.one_hot(clamped_targets, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        mask_expanded = valid_mask.unsqueeze(1).float()
        probs = probs * mask_expanded
        one_hot = one_hot * mask_expanded

        intersection = torch.sum(probs * one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + one_hot, dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        present_classes = torch.sum(one_hot, dim=(0, 2, 3)) > 0
        if present_classes.sum() > 0:
            loss = 1.0 - dice_per_class[present_classes].mean()
        else:
            loss = 1.0 - dice_per_class.mean()

        return loss


class CombinedLoss(nn.Module):
    """Baseline weighted sum of Cross-Entropy and Soft-Dice Loss (retained bit-identical)."""

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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        total_loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(
                f"Loss is NaN or Inf! CE: {loss_ce.item()}, Dice: {loss_dice.item()}"
            )
        return total_loss


class OHEMCrossEntropyLoss(nn.Module):
    """Online Hard Example Mining (OHEM) with optional class-frequency rebalancing.

    Filters pixel loss down to the top-K hardest pixels in each training step.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
        ohem_ratio: float = 0.5,
        min_kept: int = 10000,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.ohem_ratio = ohem_ratio
        self.min_kept = min_kept

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-pixel cross entropy loss: shape (B, H, W)
        weights = self.class_weights if hasattr(self, "class_weights") else None
        loss_pixel = F.cross_entropy(
            logits,
            targets,
            weight=weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid_mask = targets != self.ignore_index
        valid_loss = loss_pixel[valid_mask]

        num_valid = valid_loss.numel()
        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        k = max(int(self.ohem_ratio * num_valid), min(self.min_kept, num_valid))
        hard_loss, _ = torch.topk(valid_loss, k=k, sorted=False)
        return hard_loss.mean()


class ClassWeightedBoundaryLoss(nn.Module):
    """Mentor-proposed boundary loss focusing heavily on small/rare objects.

    Extracts 4-connected boundary pixels on-the-fly and applies a focal loss
    weighted by class-specific importance multipliers (cars, pedestrians, signs).
    """

    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 255,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma

        # Class importance multipliers beta_c (gentle multipliers to prevent gradient instability)
        # 3.0 for vehicles and humans, 2.0 for thin traffic objects, 1.0 for background
        beta = torch.ones(num_classes, dtype=torch.float32)
        # Vehicles & humans (small/rare objects with zero recall: person, rider, car, truck, bus, train, motorcycle, bicycle)
        beta[[11, 12, 13, 14, 15, 16, 17, 18]] = 3.0
        # Thin traffic objects (fence, pole, traffic light, traffic sign)
        beta[[4, 5, 6, 7]] = 2.0
        self.register_buffer("beta", beta)

    def extract_boundary(self, targets: torch.Tensor) -> torch.Tensor:
        """Extract boundary mask where neighboring valid pixel has a different class."""
        valid = targets != self.ignore_index
        boundary = torch.zeros_like(valid, dtype=torch.bool)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_t = torch.roll(torch.roll(targets, dy, dims=-2), dx, dims=-1)
            shifted_v = torch.roll(torch.roll(valid, dy, dims=-2), dx, dims=-1)
            boundary |= (valid & shifted_v & (targets != shifted_t))

        return boundary

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        boundary_mask = self.extract_boundary(targets)

        if not boundary_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Softmax probabilities: (B, C, H, W)
        probs = F.softmax(logits, dim=1)

        # Gather probability of true target class at boundary pixels
        b_targets = targets[boundary_mask]  # (N_bound,)
        # Reshape logits / probs to (N_bound, C)
        probs_b = probs.permute(0, 2, 3, 1)[boundary_mask]  # (N_bound, C)

        p_true = probs_b.gather(1, b_targets.unsqueeze(1)).squeeze(1)  # (N_bound,)
        p_true = torch.clamp(p_true, min=1e-7, max=1.0)

        # Class weight multipliers
        w_b = self.beta[b_targets]  # (N_bound,)

        # Focal boundary penalty: w_b * (1 - p_true)^gamma * (-log(p_true))
        focal_weight = torch.pow(1.0 - p_true, self.focal_gamma)
        boundary_loss = w_b * focal_weight * (-torch.log(p_true))

        return boundary_loss.sum() / (w_b.sum() + 1e-7)


class BOLTLoss(nn.Module):
    """Unified Novel Training Objective (BOLT-Loss).

    L_total = ce_weight * L_OHEM-wCE + dice_weight * L_Dice + boundary_weight * L_bound
    Operates at 0 inference overhead (exact same model at test time).
    """

    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
        ohem_ratio: float = 0.5,
        min_kept: int = 10000,
        use_class_weights: bool = True,
        use_ohem: bool = True,
        use_boundary_loss: bool = True,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.use_ohem = use_ohem
        self.use_boundary_loss = use_boundary_loss

        class_weights = get_acdc_class_weights(c_smooth=1.02, num_classes=num_classes) if use_class_weights else None

        if use_ohem:
            self.ce_loss = OHEMCrossEntropyLoss(
                ignore_index=ignore_index,
                class_weights=class_weights,
                ohem_ratio=ohem_ratio,
                min_kept=min_kept,
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index,
            )

        self.dice_loss = SoftDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        if use_boundary_loss:
            self.boundary_loss = ClassWeightedBoundaryLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
        else:
            self.boundary_loss = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        total_loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        if self.use_boundary_loss and self.boundary_loss is not None:
            loss_bound = self.boundary_loss(logits, targets)
            total_loss = total_loss + self.boundary_weight * loss_bound

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError("BOLTLoss produced NaN or Inf!")

        return total_loss


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    """Build loss function based on YAML configuration.

    Supports:
    - 'baseline' (or default): Standard CombinedLoss (CE + SoftDice)
    - 'bolt': BOLTLoss (Class-Weighted OHEM + SoftDice + Class-Weighted Boundary Loss)
    """
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})
    num_classes = ds_cfg.get("num_classes", 19)
    ignore_index = ds_cfg.get("ignore_index", 255)

    loss_type = train_cfg.get("loss_type", "baseline").lower()

    if loss_type == "rcs":
        # Step 2: Mild Class-Weighted CE + Soft-Dice
        mild_weights = get_rcs_mild_class_weights()
        boundary_weight = train_cfg.get("boundary_weight", 0.0)
        if boundary_weight > 0:
            return BOLTLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
                ce_weight=train_cfg.get("ce_weight", 1.0),
                dice_weight=train_cfg.get("dice_weight", 1.0),
                boundary_weight=boundary_weight,
                use_class_weights=True,
                use_ohem=False,
                use_boundary_loss=True,
            )
        else:
            return CombinedLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
                ce_weight=train_cfg.get("ce_weight", 1.0),
                dice_weight=train_cfg.get("dice_weight", 1.0),
                class_weights=mild_weights,
            )

    if loss_type in ["bolt", "boundary"]:
        return BOLTLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            ce_weight=train_cfg.get("ce_weight", 1.0),
            dice_weight=train_cfg.get("dice_weight", 1.0),
            boundary_weight=train_cfg.get("boundary_weight", 0.2),
            ohem_ratio=train_cfg.get("ohem_ratio", 0.5),
            min_kept=train_cfg.get("min_kept", 10000),
            use_class_weights=train_cfg.get("use_class_weights", False),
            use_ohem=train_cfg.get("use_ohem", False),
            use_boundary_loss=train_cfg.get("use_boundary_loss", True),
        )
    else:
        # Default baseline path: strictly bit-identical
        class_weights = train_cfg.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return CombinedLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            ce_weight=train_cfg.get("ce_weight", 1.0),
            dice_weight=train_cfg.get("dice_weight", 1.0),
            class_weights=class_weights,
        )
