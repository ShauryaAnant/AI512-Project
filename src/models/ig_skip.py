"""Illumination-Gated Skip Modulation (IG-Skip) Module for Nighttime Vision.

Technical Justification:
Under nighttime driving conditions, extreme localized glare from streetlights and high-beam
headlights contrasts with pitch-black shadows. Raw early encoder features (t1..t5) contain
noisy photometric glare halos. Blindly adding them (out + t) injects noise into decoder
representations, causing thin/small objects (traffic lights, poles, pedestrians) to be
subsumed by background buildings or road.

IG-Skip extracts a parameter-free CIE luminance prior L(X) = 0.299R + 0.587G + 0.114B
and passes it through a lightweight pointwise gate and depthwise edge recovery convolution:
    G_k = sigmoid(Conv1x1(L_k))
    t_k_gated = t_k * G_k
    t_k_edge = DWConv3x3(t_k * (1 - G_k))
    t_k_tilde = t_k_gated + t_k_edge
    Decoder_in = Decoder_feat + t_k_tilde

Adds only ~1,162 parameters total across all 5 decoder stages (<0.0012M), preserving the
ultra-lightweight real-time profile (<0.03M total parameters).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model import MCHLayer, trunc_normal_


class IlluminationGatedSkip(nn.Module):
    """Pointwise illumination gate with depthwise structural edge recovery."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Pointwise gate: Conv 1x1 on luminance prior (2 params)
        self.gate_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        # Depthwise 3x3 conv for structural edge extraction in poorly lit regions (9 * C params)
        self.edge_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        # Initialize gate to pass signal smoothly (bias = 0.5, weight = 1.0)
        nn.init.constant_(self.gate_conv.weight, 1.0)
        nn.init.constant_(self.gate_conv.bias, 0.0)

        # Initialize edge conv with small weights
        nn.init.normal_(self.edge_conv.weight, std=0.01)

    def forward(self, skip_feat: torch.Tensor, luminance: torch.Tensor) -> torch.Tensor:
        """Modulate skip connection with illumination prior.

        Args:
            skip_feat: Encoder feature map (B, C_k, H_k, W_k).
            luminance: Full-resolution or downsampled luminance map (B, 1, H, W).

        Returns:
            Modulated skip feature (B, C_k, H_k, W_k).
        """
        target_size = (skip_feat.size(2), skip_feat.size(3))

        # Downsample luminance to stage resolution
        if luminance.shape[2:] != target_size:
            lum_k = F.interpolate(luminance, size=target_size, mode="bilinear", align_corners=False)
        else:
            lum_k = luminance

        # Illumination confidence gate G_k in (0, 1)
        gate = torch.sigmoid(self.gate_conv(lum_k))  # (B, 1, H_k, W_k)

        # Gated skip: well-lit contrast regions pass through directly
        t_gated = skip_feat * gate

        # Edge recovery: poorly lit / shadow regions undergo depthwise structural filtering
        t_edge = self.edge_conv(skip_feat * (1.0 - gate))

        return t_gated + t_edge


class ModelIGSkip(nn.Module):
    """Mamba/SSM U-Net with Illumination-Gated Skip Modulation (IG-Skip).

    Preserves the exact backbone architecture and parameter count of the baseline,
    augmenting only the 5 skip fusion junctions with ~1,162 parameters total.
    """

    def __init__(
        self,
        num_classes: int = 19,
        input_channels: int = 3,
        c_list: Optional[List[int]] = None,
        return_logits: bool = True,
    ) -> None:
        super().__init__()
        if c_list is None:
            c_list = [8, 16, 24, 32, 48, 64]

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.c_list = c_list
        self.return_logits = return_logits

        # Encoders (Identical to baseline)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        )
        self.encoder4 = nn.Sequential(
            MCHLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            MCHLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            MCHLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        # Decoders (Identical to baseline)
        self.decoder1 = nn.Sequential(
            MCHLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            MCHLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            MCHLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)
        )

        # GroupNorm layers (Identical to baseline)
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        # Novelty: 5 Illumination-Gated Skip Modulation Blocks
        self.ig_skip5 = IlluminationGatedSkip(c_list[4])  # 48 channels
        self.ig_skip4 = IlluminationGatedSkip(c_list[3])  # 32 channels
        self.ig_skip3 = IlluminationGatedSkip(c_list[2])  # 24 channels
        self.ig_skip2 = IlluminationGatedSkip(c_list[1])  # 16 channels
        self.ig_skip1 = IlluminationGatedSkip(c_list[0])  # 8 channels

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def extract_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """Extract parameter-free CIE luminance prior L(X) in [0, 1].

        Handles ImageNet normalized inputs:
        mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        """
        r = torch.clamp(x[:, 0:1] * 0.229 + 0.485, 0.0, 1.0)
        g = torch.clamp(x[:, 1:2] * 0.224 + 0.456, 0.0, 1.0)
        b = torch.clamp(x[:, 2:3] * 0.225 + 0.406, 0.0, 1.0)
        # CIE standard luminance
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return lum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Illumination-Gated Skip Modulation.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Output logits of shape (B, num_classes, H, W).
        """
        # Step 1: Parameter-free physical luminance extraction
        lum = self.extract_luminance(x)

        # Encoder path
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # (B, c0, H/2, W/2)

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # (B, c1, H/4, W/4)

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # (B, c2, H/8, W/8)

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # (B, c3, H/16, W/16)

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # (B, c4, H/32, W/32)

        # Bottleneck
        out = F.gelu(self.encoder6(out))  # (B, c5, H/32, W/32)

        # Decoder path with IG-Skip Modulation
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        t5_mod = self.ig_skip5(t5, lum)
        out5 = torch.add(out5, t5_mod)

        out4 = F.gelu(
            F.interpolate(
                self.dbn2(self.decoder2(out5)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        t4_mod = self.ig_skip4(t4, lum)
        out4 = torch.add(out4, t4_mod)

        out3 = F.gelu(
            F.interpolate(
                self.dbn3(self.decoder3(out4)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        t3_mod = self.ig_skip3(t3, lum)
        out3 = torch.add(out3, t3_mod)

        out2 = F.gelu(
            F.interpolate(
                self.dbn4(self.decoder4(out3)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        t2_mod = self.ig_skip2(t2, lum)
        out2 = torch.add(out2, t2_mod)

        out1 = F.gelu(
            F.interpolate(
                self.dbn5(self.decoder5(out2)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        t1_mod = self.ig_skip1(t1, lum)
        out1 = torch.add(out1, t1_mod)

        out0 = F.interpolate(
            self.final(out1),
            scale_factor=(2, 2),
            mode="bilinear",
            align_corners=True,
        )

        if not self.return_logits:
            return torch.sigmoid(out0)
        return out0
