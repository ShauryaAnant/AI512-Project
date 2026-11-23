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

    def __init__(self, model_name: str, device: torch.device, gamma_brighten: float = 0.6) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.gamma_brighten = gamma_brighten

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
        # If gamma brighten is set, brighten image for the daylight-trained teacher
        if self.gamma_brighten > 0:
            # Denormalize ImageNet
            r = torch.clamp(images[:, 0:1] * 0.229 + 0.485, 0.0, 1.0)
            g = torch.clamp(images[:, 1:2] * 0.224 + 0.456, 0.0, 1.0)
            b = torch.clamp(images[:, 2:3] * 0.225 + 0.406, 0.0, 1.0)
            rgb = torch.cat([r, g, b], dim=1)
            # Apply gamma brightening (gamma = 0.6 lifts night shadows)
            rgb_bright = torch.pow(rgb, self.gamma_brighten)
            # Renormalize ImageNet for SegFormer
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            teacher_input = (rgb_bright - mean) / std
        else:
            teacher_input = images

        # SegFormer outputs logits at 1/4 resolution (B, 19, H/4, W/4)
        outputs = self.model(pixel_values=teacher_input)
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
    gamma_brighten = kd_cfg.get("gamma_brighten", 0.6)

    # Offline/test mode support
    if teacher_name.lower() == "dummy":
        return DummyTeacher(num_classes=cfg.get("dataset", {}).get("num_classes", 19)).to(device)

    return SegFormerTeacher(model_name=teacher_name, device=device, gamma_brighten=gamma_brighten)
