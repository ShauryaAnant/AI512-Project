"""Main training script for ACDC Night semantic segmentation.

Supports config-driven execution, AdamW with lr=0.01, CosineAnnealingLR,
per-epoch evaluation, checkpointing (last.pth and best.pth), and auto-resume.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import time
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets.acdc import build_dataloader
from src.engine import evaluate, train_one_epoch
from src.losses import CombinedLoss
from src.models.model import build_model
from src.utils.ckpt import load_checkpoint, save_checkpoint
from src.utils.flops import compute_model_flops, count_parameters
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def train(cfg: dict, resume: bool = False):
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    out_dir = Path(train_cfg.get("out_dir", "runs/baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(out_dir), name="train")
    logger.info("=== Starting Training Pipeline ===")
    logger.info(f"Output directory: {out_dir}")

    # Set seed
    seed = train_cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    # Hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Data loaders
    data_root = ds_cfg.get("data_root", "dataset")
    splits_dir = ds_cfg.get("splits_dir", "splits")
    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 2)
    crop_size = tuple(ds_cfg.get("crop_size", [512, 512]))
    eval_size = tuple(ds_cfg.get("eval_size", [512, 1024]))

    train_loader = build_dataloader(
        split="train",
        data_root=data_root,
        splits_dir=splits_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        crop_size=crop_size,
        eval_size=eval_size,
    )

    test_loader = build_dataloader(
        split="test",
        data_root=data_root,
        splits_dir=splits_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        crop_size=crop_size,
        eval_size=eval_size,
    )
    logger.info(f"Loaded {len(train_loader.dataset)} training images and {len(test_loader.dataset)} test images.")

    # Instantiate Model
    model = build_model(cfg=cfg)
    model.to(device)

    param_info = count_parameters(model)
    logger.info(f"Model parameters: {param_info['params_total_M']:.4f}M ({param_info['params_total']:,})")

    # Loss function
    criterion = CombinedLoss(
        num_classes=ds_cfg.get("num_classes", 19),
        ignore_index=ds_cfg.get("ignore_index", 255),
        ce_weight=train_cfg.get("ce_weight", 1.0),
        dice_weight=train_cfg.get("dice_weight", 1.0),
    ).to(device)

    # Optimizer & Scheduler
    lr = train_cfg.get("lr", 0.01)
    min_lr = train_cfg.get("min_lr", 1e-6)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    epochs = train_cfg.get("epochs", 300)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    start_epoch = 0
    best_test_miou = 0.0

    # Resume support
    last_ckpt_path = out_dir / "checkpoints" / "last.pth"
    if resume and last_ckpt_path.exists():
        logger.info(f"Resuming from checkpoint: {last_ckpt_path}")
        ckpt = load_checkpoint(str(last_ckpt_path), model, optimizer, scheduler, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_test_miou = ckpt.get("best_test_miou", 0.0)
        logger.info(f"Resumed successfully at epoch {start_epoch} (Best mIoU so far: {best_test_miou * 100:.2f}%)")

    # Metrics CSV tracking
    metrics_csv = out_dir / "metrics.csv"
    csv_exists = metrics_csv.exists()
    csv_file = open(metrics_csv, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "lr", "train_loss", "grad_norm", "test_loss", "test_mIoU", "test_pixel_acc"])
        csv_file.flush()

    # Training loop
    logger.info(f"Beginning training from epoch {start_epoch + 1} to {epochs}...")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        train_loss, grad_norm = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_grad=train_cfg.get("clip_grad", 1.0),
            logger=logger,
        )
        scheduler.step()

        # Evaluate on test set
        test_metrics, test_loss = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        test_miou = test_metrics["mIoU"]
        test_pix_acc = test_metrics["pixel_acc"]
        elapsed = time.time() - t0

        is_best = test_miou > best_test_miou
        if is_best:
            best_test_miou = test_miou

        logger.info(
            f"Epoch [{epoch + 1:03d}/{epochs:03d}] "
            f"Loss: {train_loss:.4f} | Test mIoU: {test_miou * 100:.2f}% (Best: {best_test_miou * 100:.2f}%) | "
            f"PixAcc: {test_pix_acc * 100:.2f}% | LR: {curr_lr:.6f} | Time: {elapsed:.1f}s"
        )

        # Log to CSV
        csv_writer.writerow([epoch + 1, curr_lr, train_loss, grad_norm, test_loss, test_miou, test_pix_acc])
        csv_file.flush()

        # Save checkpoint
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_test_miou": best_test_miou,
            "config": cfg,
        }
        save_checkpoint(state, is_best=is_best, out_dir=str(out_dir))

    csv_file.close()
    logger.info(f"Training complete! Best Test mIoU: {best_test_miou * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train semantic segmentation model")
    parser.add_argument("--config", type=str, default="configs/baseline_acdc_night.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.data_root is not None:
        config["dataset"]["data_root"] = args.data_root
    if args.out_dir is not None:
        config["training"]["out_dir"] = args.out_dir

    train(config, resume=args.resume)
