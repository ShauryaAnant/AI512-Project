"""Teacher Model Loader & Wrapper for SegFormer-based Knowledge Distillation.

Loads official public pretrained SegFormer models (e.g. SegFormer-B2 on Cityscapes)
via Hugging Face `transformers`, freezes all parameters, and wraps inference with
automatic bilinear interpolation from 1/4 resolution to full crop size.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerTeacher(nn.Module):
    """Frozen SegFormer teacher producing 19-class logits at input resolution."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device

        try:
            from transformers import SegformerForSemanticSegmentation
            print(f"[Distillation] Loading public teacher model: '{model_name}'...")
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load teacher '{model_name}' via transformers. "
                f"Ensure `transformers` is installed and internet is accessible. Error: {e}"
            )

        self.model.to(device)
        self.model.eval()

        # Strictly freeze all teacher parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen teacher.

        Args:
            images: (B, 3, H, W) normalized input images.

        Returns:
            logits: (B, 19, H, W) bilinearly upsampled teacher logits.
        """
        # SegFormer outputs logits at 1/4 resolution (B, 19, H/4, W/4)
        outputs = self.model(pixel_values=images)
        logits = outputs.logits

        # Upsample to full input resolution
        target_size = (images.size(2), images.size(3))
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


class DummyTeacher(nn.Module):
    """Fallback dummy teacher for offline unit testing without downloading 100MB+ weights."""

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        # Return smooth random logits
        return torch.randn(b, self.num_classes, h, w, device=images.device)


def build_teacher(cfg: Dict[str, Any], device: torch.device) -> Optional[nn.Module]:
    """Factory function to build teacher model if distillation is enabled.

    Args:
        cfg: Overall experiment configuration dict.
        device: Torch device (cuda/cpu).

    Returns:
        SegFormerTeacher instance if enabled, else None.
    """
    kd_cfg = cfg.get("distillation", {})
    if not kd_cfg.get("enabled", False):
        return None

    teacher_name = kd_cfg.get("teacher_name", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")

    # Offline/test mode support
    if teacher_name.lower() == "dummy":
        return DummyTeacher(num_classes=cfg.get("dataset", {}).get("num_classes", 19)).to(device)

    return SegFormerTeacher(model_name=teacher_name, device=device)
