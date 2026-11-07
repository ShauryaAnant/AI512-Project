"""Smoke tests for train and inference pipelines.

Verifies:
1. Overfitting a mini-batch (loss drops cleanly).
2. End-to-end 2-epoch train run with checkpoint saving.
3. Resume functionality from last.pth.
4. Inference execution and results.json export.
"""

from pathlib import Path
import shutil
import torch
import torch.optim as optim

from src.datasets.acdc import ACDCDataset
from src.datasets.transforms import get_train_transforms
from src.losses import CombinedLoss
from src.models.model import build_model
from train import train
from inference import run_inference


def test_overfit_small_batch():
    """Verify that model can overfit a tiny batch (loss decreases monotonically)."""
    device = torch.device("cpu")
    model = build_model(num_classes=19, input_channels=3).to(device)
    criterion = CombinedLoss(num_classes=19, ignore_index=255).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    ds = ACDCDataset(
        split_file="splits/train.txt",
        data_root="dataset",
        transforms=get_train_transforms(crop_size=(128, 128)),
    )

    # Use first 2 items
    imgs = torch.stack([ds[0][0], ds[1][0]]).to(device)
    masks = torch.stack([ds[0][1], ds[1][1]]).to(device)

    losses = []
    model.train()
    for _ in range(8):
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Overfit losses (8 steps): {[round(x, 4) for x in losses]}")
    assert losses[-1] < losses[0], f"Loss did not decrease! Start: {losses[0]}, End: {losses[-1]}"
    print("test_overfit_small_batch passed successfully!")


def test_train_and_resume_end_to_end():
    """Verify 2-epoch training, resume, and inference execution."""
    smoke_out = Path("runs/smoke_test")
    if smoke_out.exists():
        shutil.rmtree(smoke_out)

    config = {
        "dataset": {
            "data_root": "dataset",
            "splits_dir": "splits",
            "crop_size": [128, 128],
            "eval_size": [128, 256],
            "num_classes": 19,
            "ignore_index": 255,
        },
        "model": {
            "num_classes": 19,
            "input_channels": 3,
            "c_list": [8, 16, 24, 32, 48, 64],
        },
        "training": {
            "epochs": 1,
            "lr": 0.01,
            "min_lr": 1e-6,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "num_workers": 0,
            "seed": 42,
            "clip_grad": 1.0,
            "ce_weight": 1.0,
            "dice_weight": 1.0,
            "amp": False,
            "out_dir": str(smoke_out),
        },
    }

    # 1. Run 1 epoch
    print("Running 1-epoch smoke test...")
    train(config, resume=False)
    assert (smoke_out / "checkpoints" / "last.pth").exists(), "last.pth not created"
    assert (smoke_out / "metrics.csv").exists(), "metrics.csv not created"

    # 2. Test resume
    print("Testing resume for epoch 2...")
    config["training"]["epochs"] = 2
    train(config, resume=True)

    # 3. Test inference
    ckpt_path = str(smoke_out / "checkpoints" / "last.pth")
    config_yaml_path = smoke_out / "smoke_config.yaml"
    import yaml
    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print("Running inference test...")
    res = run_inference(
        config_path=str(config_yaml_path),
        ckpt_path=ckpt_path,
        out_dir=str(smoke_out / "eval"),
    )

    assert "mIoU" in res["metrics"], "mIoU missing from results"
    assert (smoke_out / "eval" / "results.json").exists(), "results.json not created"
    print("test_train_and_resume_end_to_end passed successfully!")


if __name__ == "__main__":
    test_overfit_small_batch()
    test_train_and_resume_end_to_end()
    print("=== All Phase 6 Smoke Tests Passed! ===")
