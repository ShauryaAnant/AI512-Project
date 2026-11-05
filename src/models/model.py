"""Supplied U-Net model with Mamba/VSS hybrid blocks for semantic segmentation.

Adapted for 19-class ACDC night semantic segmentation.
Outputs raw logits of shape (B, num_classes, H, W).
"""

from typing import Any, Dict, List, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_


class CustomMamba(nn.Module):
    """Custom pure-PyTorch implementation of Mamba selective state space layer.

    Supports both vectorized parallel prefix scan and sequential scan.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 2,
        expand: int = 1,
        dt_rank: Any = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: splits into main branch (x) and gate (z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Depthwise causal conv over sequence axis
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # Project x -> (Delta_rank, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Low-rank Delta -> full d_inner Delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

        # Diagonal state matrix A in log-space (strictly negative)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))  # (d_inner, d_state)

        # Skip / feedthrough term
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, L, d_model).

        Returns:
            Output tensor of shape (B, L, d_model).
        """
        b, l, _ = x.shape

        # Split into main path (x_in) and gating path (z)
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :l]
        x_in = F.silu(x_in)
        x_in = x_in.transpose(1, 2)

        # Input-dependent Delta, B, C
        dbc = self.x_proj(x_in)
        dt, b_mat, c_mat = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))

        y = self._selective_scan_parallel(x_in, delta, b_mat, c_mat)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan_parallel(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        b_mat: torch.Tensor,
        c_mat: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel prefix scan (Hillis-Steele doubling) for SSM recurrence.

        Args:
            x: (B, L, d_inner)
            delta: (B, L, d_inner)
            b_mat: (B, L, d_state)
            c_mat: (B, L, d_state)

        Returns:
            (B, L, d_inner)
        """
        _, l, _ = x.shape
        a = -torch.exp(self.A_log)

        # per-step affine map: h_t = a_t * h_{t-1} + b_t
        a_map = torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))
        b_map = delta.unsqueeze(-1) * b_mat.unsqueeze(2) * x.unsqueeze(-1)

        shift = 1
        while shift < l:
            a_shift = F.pad(a_map[:, :-shift], (0, 0, 0, 0, shift, 0), value=1.0)
            b_shift = F.pad(b_map[:, :-shift], (0, 0, 0, 0, shift, 0), value=0.0)

            b_map = torch.addcmul(b_map, a_map, b_shift)
            a_map = a_map * a_shift
            shift *= 2

        state = b_map
        y = torch.einsum("bldn,bln->bld", state, c_mat)
        y = y + x * self.D
        return y


class MCHLayer(nn.Module):
    """Multi-Channel Hybrid Layer.

    Splits features into 4 branches:
    1. Identity
    2. Local 3x3 depthwise conv
    3. Global CustomMamba branch
    4. Dilated 3x3 depthwise conv (dilation=2)
    Fuses via concatenation, LayerNorm, and linear projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_state: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.norm = nn.LayerNorm(input_dim)
        branch_dim = input_dim // 4

        self.mamba = CustomMamba(
            d_model=branch_dim,
            d_state=d_state,
            d_conv=2,
            expand=1,
        )

        self.dw = nn.Conv2d(
            branch_dim, branch_dim, 3, padding=1, groups=branch_dim
        )
        self.dw_dil = nn.Conv2d(
            branch_dim, branch_dim, 3, padding=2, dilation=2, groups=branch_dim
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Output tensor (B, output_dim, H, W).
        """
        if x.dtype == torch.float16:
            x = x.float()

        b, c, h, w = x.shape
        assert c == self.input_dim, f"Input channels {c} != expected {self.input_dim}"
        n = h * w

        x_flat = x.reshape(b, c, n).transpose(1, 2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        def to_2d(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).reshape(b, c // 4, h, w)

        def to_seq(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, c // 4, n).transpose(1, 2)

        b1 = x1
        b2 = to_seq(self.dw(to_2d(x2)))
        b3 = self.mamba(x3) + self.skip_scale * x3
        b4 = to_seq(self.dw_dil(to_2d(x4)))

        out = torch.cat([b1, b2, b3, b4], dim=2)
        out = self.norm(out)
        out = self.proj(out)

        return out.transpose(1, 2).reshape(b, self.output_dim, h, w)


class Model(nn.Module):
    """VM-UNet style lightweight semantic segmentation model.

    6 encoder stages and 5 decoder stages with skip connections.
    Outputs raw logits of shape (B, num_classes, H, W).
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

        # Encoders
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

        # Decoders
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

        # GroupNorm layers
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

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, input_channels, H, W).
               Height and width must be divisible by 32.

        Returns:
            Output logits of shape (B, num_classes, H, W).
        """
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

        # Decoder path with skip connections
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)

        out4 = F.gelu(
            F.interpolate(
                self.dbn2(self.decoder2(out5)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        out4 = torch.add(out4, t4)

        out3 = F.gelu(
            F.interpolate(
                self.dbn3(self.decoder3(out4)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        out3 = torch.add(out3, t3)

        out2 = F.gelu(
            F.interpolate(
                self.dbn4(self.decoder4(out3)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        out2 = torch.add(out2, t2)

        out1 = F.gelu(
            F.interpolate(
                self.dbn5(self.decoder5(out2)),
                scale_factor=(2, 2),
                mode="bilinear",
                align_corners=True,
            )
        )
        out1 = torch.add(out1, t1)

        out0 = F.interpolate(
            self.final(out1),
            scale_factor=(2, 2),
            mode="bilinear",
            align_corners=True,
        )

        if not self.return_logits:
            return torch.sigmoid(out0)
        return out0


def build_model(
    cfg: Optional[Dict[str, Any]] = None,
    num_classes: int = 19,
    input_channels: int = 3,
    **kwargs: Any,
) -> Model:
    """Build and return semantic segmentation model instance.

    Args:
        cfg: Optional configuration dictionary.
        num_classes: Number of semantic output classes (19 for ACDC / Cityscapes).
        input_channels: Number of input image channels (default 3).
        **kwargs: Additional parameters passed to Model.

    Returns:
        Instantiated Model.
    """
    if cfg is not None:
        model_cfg = cfg.get("model", {})
        num_classes = model_cfg.get("num_classes", num_classes)
        input_channels = model_cfg.get("input_channels", input_channels)
        c_list = model_cfg.get("c_list", None)
        if c_list is not None:
            kwargs["c_list"] = c_list

    return Model(
        num_classes=num_classes,
        input_channels=input_channels,
        return_logits=True,
        **kwargs,
    )
