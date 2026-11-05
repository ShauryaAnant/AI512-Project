"""Unit tests for Model architecture and shape contracts."""

try:
    import pytest
except ImportError:
    pytest = None
import torch
from src.models.model import Model, build_model


def test_model_forward_and_backward():
    """Verify forward pass output shape and backward gradient flow."""
    model = build_model(num_classes=19, input_channels=3)
    model.train()

    # Input tensor divisible by 32 (e.g. 64x64)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    out = model(x)

    assert out.shape == (2, 19, 64, 64), f"Expected (2, 19, 64, 64), got {out.shape}"
    assert not torch.isnan(out).any(), "Model output contains NaNs"

    # Backward pass
    target = torch.randint(0, 19, (2, 64, 64))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, target)
    loss.backward()

    assert not torch.isnan(loss), "Loss is NaN"
    assert x.grad is not None, "Input gradients not computed"
    print("test_model_forward_and_backward passed successfully!")


def test_model_resolutions():
    """Verify forward at multiple resolutions divisible by 32."""
    model = build_model(num_classes=19, input_channels=3)
    model.eval()

    test_shapes = [(1, 3, 32, 32), (1, 3, 64, 128), (1, 3, 128, 128)]

    with torch.no_grad():
        for b, c, h, w in test_shapes:
            x = torch.randn(b, c, h, w)
            out = model(x)
            assert out.shape == (b, 19, h, w), f"Expected shape ({b}, 19, {h}, {w}), got {out.shape}"
    print("test_model_resolutions passed successfully!")


def test_parameter_count():
    """Compute and display model parameter count."""
    model = build_model(num_classes=19, input_channels=3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.4f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.4f}M)")

    # Assert model is truly lightweight (< 1M parameters)
    assert total_params < 1_000_000, f"Model has {total_params} parameters, expected < 1M"


if __name__ == "__main__":
    test_model_forward_and_backward()
    test_model_resolutions()
    test_parameter_count()
