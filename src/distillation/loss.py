"""Knowledge Distillation Loss with Nighttime Confidence Masking (NightKD).

Transfers rich inter-class semantic correlation ("dark knowledge") from a large,
well-trained teacher (e.g. SegFormer-B2 on Cityscapes) to the ultra-lightweight
27k Mamba/SSM student network.

Under nighttime conditions, deep shadows and blinding glare induce localized
teacher uncertainty. We incorporate Teacher Confidence Masking:
    w_T(x, y) = I(max_c p_T(c, x, y) >= tau)
In confident regions, the student mimics the teacher's temperature-scaled soft probabilities.
In uncertain / noisy glare regions, the loss falls back gracefully to pure ground truth.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """Teacher-Student Distillation Loss with Temperature Scaling and Confidence Gating."""

    def __init__(
        self,
        base_criterion: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.05,
        conf_threshold: float = 0.6,
        use_confidence_masking: bool = True,
        ignore_index: int = 255,
    ) -> None:
        """Initialize KnowledgeDistillationLoss.

        Args:
            base_criterion: Student task loss (e.g. CombinedLoss or SoftDice + CE).
            temperature: Softmax smoothing temperature (default 2.0).
            alpha: Balancing weight: (1 - alpha) * L_task + alpha * L_KD (default 0.05).
            conf_threshold: Minimum teacher maximum class probability to apply KD (default 0.6).
            use_confidence_masking: Whether to filter out low-confidence teacher pixels.
            ignore_index: Ground truth ignore index (255).
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.temperature = temperature
        self.alpha = alpha
        self.conf_threshold = conf_threshold
        self.use_confidence_masking = use_confidence_masking
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined Task + KD loss.

        Args:
            student_logits: (B, C, H, W) predictions from student model.
            targets: (B, H, W) ground-truth semantic mask.
            teacher_logits: (B, C, H, W) predictions from frozen teacher model.

        Returns:
            Scalar loss tensor.
        """
        # Step 1: Base ground-truth task loss (CE + SoftDice)
        loss_task = self.base_criterion(student_logits, targets)

        # If no teacher logits provided, fall back to pure task loss
        if teacher_logits is None or self.alpha <= 0.0:
            return loss_task

        # Ensure spatial resolutions match
        if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        T = self.temperature

        # Step 2: Soft probabilities with temperature
        p_teacher = F.softmax(teacher_logits / T, dim=1)
        log_p_student = F.log_softmax(student_logits / T, dim=1)

        # Pixelwise KL-Divergence: sum_c p_T * (log p_T - log p_S)
        kl_pixel = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=1)  # (B, H, W)

        # Step 3: High Confidence Masking
        valid_mask = (targets != self.ignore_index)  # (B, H, W)

        if self.use_confidence_masking:
            # Measure teacher maximum class confidence at T=1.0
            p_teacher_raw = F.softmax(teacher_logits, dim=1)
            teacher_conf = p_teacher_raw.max(dim=1).values  # (B, H, W)
            kd_mask = valid_mask & (teacher_conf >= self.conf_threshold)
        else:
            kd_mask = valid_mask

        num_kd_pixels = kd_mask.float().sum()
        if num_kd_pixels > 0:
            # Normalized unscaled KL loss (keeps loss around 0.5 - 1.5 instead of exploding to 16x)
            loss_kd = (kl_pixel * kd_mask.float()).sum() / num_kd_pixels
        else:
            loss_kd = torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        total_loss = (1.0 - self.alpha) * loss_task + self.alpha * loss_kd
        return total_loss
