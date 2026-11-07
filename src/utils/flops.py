"""FLOPs and parameter counting utilities.

Accurately calculates parameter counts and FLOPs/MACs, including
standard convolutional/linear layers and CustomMamba selective-scan recurrence.
"""

from typing import Dict, Tuple
import math
import numpy as np
import torch
import torch.nn as nn

from src.models.model import CustomMamba


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dict with total_params and trainable_params (both in millions).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total_M": float(total / 1e6),
        "params_trainable_M": float(trainable / 1e6),
        "params_total": int(total),
        "params_trainable": int(trainable),
    }


def compute_model_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 1024),
) -> float:
    """Compute total FLOPs for the model including CustomMamba scan ops.

    Args:
        model: PyTorch model.
        input_shape: (B, C, H, W) tuple.

    Returns:
        Total estimated GFLOPs.
    """
    total_flops = 0
    hooks = []

    def conv2d_hook(m, inp, outp):
        nonlocal total_flops
        # outp: (B, C_out, H_out, W_out)
        b, c_out, h_out, w_out = outp.shape
        c_in = m.in_channels
        kh, kw = m.kernel_size
        groups = m.groups
        # FLOPs = 2 * H * W * (C_in / groups * Kh * Kw) * C_out
        flops = 2 * b * h_out * w_out * (c_in // groups) * kh * kw * c_out
        if m.bias is not None:
            flops += b * h_out * w_out * c_out
        total_flops += flops

    def conv1d_hook(m, inp, outp):
        nonlocal total_flops
        b, c_out, l_out = outp.shape
        c_in = m.in_channels
        k = m.kernel_size[0]
        groups = m.groups
        flops = 2 * b * l_out * (c_in // groups) * k * c_out
        if m.bias is not None:
            flops += b * l_out * c_out
        total_flops += flops

    def linear_hook(m, inp, outp):
        nonlocal total_flops
        # inp[0]: (B, ..., in_features)
        num_elements = inp[0].numel() // m.in_features
        flops = 2 * num_elements * m.in_features * m.out_features
        if m.bias is not None:
            flops += num_elements * m.out_features
        total_flops += flops

    def mamba_hook(m, inp, outp):
        nonlocal total_flops
        # inp[0]: (B, L, d_model)
        b, l, _ = inp[0].shape
        d_inner = m.d_inner
        n = m.d_state
        # Selective scan FLOPs (prefix doubling scan):
        # Discretization: ~4 * B * L * d_inner * N
        # Doubling scan: ceil(log2(L)) * 4 * B * L * d_inner * N
        # Output einsum (bldn, bln -> bld): 2 * B * L * d_inner * N
        num_steps = math.ceil(math.log2(l)) if l > 1 else 1
        scan_flops = (4 + 4 * num_steps + 2) * b * l * d_inner * n
        total_flops += scan_flops

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, CustomMamba):
            hooks.append(m.register_forward_hook(mamba_hook))

    # Run dummy forward
    model_mode = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        try:
            model(dummy_input)
        finally:
            for h in hooks:
                h.remove()
            if model_mode:
                model.train()

    gflops = total_flops / 1e9
    return gflops
