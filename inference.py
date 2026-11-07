"""Inference and evaluation script for ACDC Night semantic segmentation.

Evaluates trained checkpoint, saves predicted single-channel trainId PNGs
and colorized visualizations at original 1920x1080 resolution, and outputs results.json.
"""

import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.acdc import ACDCDataset
from src.datasets.transforms import get_test_transforms
from src.metrics import MetricTracker
from src.models.model import build_model
from src.utils.ckpt import load_checkpoint
from src.utils.flops import compute_model_flops, count_parameters
from src.utils.viz import colorize_mask


def run_inference(
    config_path: str,
    ckpt_path: str,
    save_preds: bool = True,
    save_color: bool = True,
    out_dir: str = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})
    if out_dir is None:
        out_dir = Path(train_cfg.get("out_dir", "runs/baseline")) / "eval"
    else:
        out_dir = Path(out_dir)

    preds_dir = out_dir / "preds"
    color_dir = out_dir / "color_preds"
    if save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)
    if save_color:
        color_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    # Load model & weights
    model = build_model(cfg=cfg)
    model.to(device)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # Load dataset
    data_root = ds_cfg.get("data_root", "dataset")
    splits_dir = ds_cfg.get("splits_dir", "splits")
    eval_size = tuple(ds_cfg.get("eval_size", [512, 1024]))

    dataset = ACDCDataset(
        split_file=f"{splits_dir}/test.txt",
        data_root=data_root,
        transforms=get_test_transforms(eval_size=eval_size),
        return_metadata=True,
    )
    print(f"Evaluating {len(dataset)} test images...")

    tracker = MetricTracker(num_classes=19, ignore_index=255)

    with torch.no_grad():
        for i in range(len(dataset)):
            img_tensor, mask_tensor, meta = dataset[i]
            # (3, H, W) -> (1, 3, H, W)
            input_tensor = img_tensor.unsqueeze(0).to(device)

            logits = model(input_tensor)

            # Upsample logits to original 1920x1080 resolution
            logits_orig = F.interpolate(
                logits,
                size=(1080, 1920),
                mode="bilinear",
                align_corners=True,
            )

            pred = torch.argmax(logits_orig, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Update metrics
            if mask_tensor is not None:
                mask_np = mask_tensor.numpy()
                if mask_np.shape != (1080, 1920) and meta.get("gt_path") and Path(meta["gt_path"]).exists():
                    with Image.open(meta["gt_path"]) as raw_m:
                        gt_orig = np.array(raw_m)
                else:
                    gt_orig = mask_np
                tracker.update(pred, gt_orig)

            stem = meta["stem"]

            if save_preds:
                pred_img = Image.fromarray(pred)
                pred_img.save(preds_dir / f"{stem}.png")

            if save_color:
                color_img = colorize_mask(pred)
                color_img.save(color_dir / f"{stem}_color.png")

            if (i + 1) % 25 == 0 or (i + 1) == len(dataset):
                print(f"Processed [{i + 1}/{len(dataset)}] images")

    metrics = tracker.compute()
    params_info = count_parameters(model)
    flops = compute_model_flops(model, input_shape=(1, 3, eval_size[0], eval_size[1]))

    # Common results schema (ROADMAP Appendix A)
    results = {
        "run_id": Path(ckpt_path).stem,
        "dataset": "acdc_night",
        "protocol": "train_400_test_106",
        "model": "baseline",
        "seed": train_cfg.get("seed", 42),
        "config": config_path,
        "checkpoint": str(ckpt_path),
        "input_resolution": list(ds_cfg.get("crop_size", [512, 512])),
        "eval_resolution": list(eval_size),
        "optimizer": "AdamW",
        "lr": train_cfg.get("lr", 0.01),
        "epochs": train_cfg.get("epochs", 300),
        "loss": "Combined_CE_SoftDice",
        "metrics": metrics,
        "efficiency": {
            "params_M": params_info["params_total_M"],
            "flops_G": float(flops),
            "flops_method": "analytical_with_custom_mamba_scan",
            "batch_size": 1,
            "device": str(device),
        },
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation Results ===")
    print(f"Test mIoU:      {metrics['mIoU'] * 100:.2f}%")
    print(f"Dice Mean:      {metrics['dice_mean'] * 100:.2f}%")
    print(f"Pixel Accuracy: {metrics['pixel_acc'] * 100:.2f}%")
    print(f"Results saved to: {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--config", type=str, default="configs/baseline_acdc_night.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_inference(args.config, args.checkpoint, out_dir=args.out_dir)
