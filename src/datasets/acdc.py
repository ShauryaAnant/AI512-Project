"""ACDC Night PyTorch Dataset and DataLoader constructor.

Loads RGB images and Cityscapes trainId annotations (0..18, 255 ignore index)
from generated split files. Supports standard joint augmentations and DAFormer-style
Rare-Class Sampling (RCS) to guarantee exposure to underrepresented classes.
"""

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.datasets.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_post_crop_transforms,
)


class ACDCDataset(Dataset):
    """PyTorch Dataset for ACDC Night semantic segmentation."""

    def __init__(
        self,
        split_file: str,
        data_root: str = "dataset",
        transforms: Optional[Callable] = None,
        return_metadata: bool = True,
        rcs_enabled: bool = False,
        rcs_prob: float = 0.5,
        rcs_stats_file: Optional[str] = "data/processed/rcs_training_stats.json",
        crop_size: Tuple[int, int] = (512, 512),
    ) -> None:
        """Initialize ACDCDataset.

        Args:
            split_file: Path to split text file.
            data_root: Root directory of ACDC dataset.
            transforms: Callable joint transform applied to (image, mask).
            return_metadata: If True, returns dict with sequence name and image stem.
            rcs_enabled: If True, activates DAFormer-style Rare-Class-Aware Crop Sampling.
            rcs_prob: Probability of sampling rare-class crops (default 0.50).
            rcs_stats_file: Path to precomputed training split class index and probabilities.
            crop_size: (H, W) crop dimensions for RCS.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.transforms = transforms
        self.return_metadata = return_metadata
        self.rcs_enabled = rcs_enabled
        self.rcs_prob = rcs_prob
        self.crop_h, self.crop_w = crop_size

        self.class_to_indices: Dict[int, List[int]] = {}
        self.sampling_probs: Optional[np.ndarray] = None
        self.post_crop_transforms = get_post_crop_transforms() if rcs_enabled else None

        # Load precomputed RCS statistics if enabled
        if self.rcs_enabled and rcs_stats_file:
            stats_path = Path(rcs_stats_file)
            if not stats_path.exists():
                # Check relative to parent directory if running in nested path
                alt_path = Path(__file__).resolve().parent.parent.parent / rcs_stats_file
                if alt_path.exists():
                    stats_path = alt_path

            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    self.sampling_probs = np.array(stats["sampling_probabilities"], dtype=np.float64)
                    self.class_to_indices = {int(k): v for k, v in stats["class_to_indices"].items()}
            else:
                print(f"Warning: RCS stats file {rcs_stats_file} not found. Falling back to standard crops.")

        # Auto-detect actual data root if nested inside subfolders (e.g. on Kaggle)
        if self.data_root.exists() and not (self.data_root / "rgb_anon" / "rgb_anon").exists():
            for candidate in self.data_root.rglob("rgb_anon"):
                if candidate.is_dir() and (candidate / "rgb_anon" / "night").exists():
                    self.data_root = candidate.parent
                    break
                elif candidate.is_dir() and (candidate / "night").exists():
                    self.data_root = candidate.parent.parent
                    break

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        self.samples: List[Dict[str, str]] = []
        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                rel_rgb = parts[0]
                rel_gt = parts[1] if len(parts) > 1 and parts[1] else None
                rel_inv = parts[2] if len(parts) > 2 and parts[2] else None
                seq_name = parts[3] if len(parts) > 3 else ""
                stem = parts[4] if len(parts) > 4 else ""

                self.samples.append(
                    {
                        "rgb_path": str(self.data_root / rel_rgb),
                        "gt_path": str(self.data_root / rel_gt) if rel_gt else None,
                        "inv_path": str(self.data_root / rel_inv) if rel_inv else None,
                        "sequence": seq_name,
                        "stem": stem,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Step 1: Check if Rare-Class Sampling should be triggered
        do_rcs = (
            self.rcs_enabled
            and (self.sampling_probs is not None)
            and (random.random() < self.rcs_prob)
            and len(self.class_to_indices) > 0
        )

        if do_rcs:
            # Sample rare class according to P(c) ~ f_c^(-0.5)
            c = int(np.random.choice(len(self.sampling_probs), p=self.sampling_probs))
            candidate_indices = self.class_to_indices.get(c, [])
            if candidate_indices:
                chosen_idx = random.choice(candidate_indices)
                item = self.samples[chosen_idx]
                rgb_path = item["rgb_path"]
                gt_path = item["gt_path"]

                try:
                    with Image.open(rgb_path) as img_pil:
                        img = img_pil.convert("RGB")
                    with Image.open(gt_path) as mask_pil:
                        mask = mask_pil.copy()

                    mask_np = np.array(mask)
                    ys, xs = np.where(mask_np == c)
                    if len(ys) > 0:
                        k = random.randint(0, len(ys) - 1)
                        cy, cx = ys[k], xs[k]
                        w, h = img.size

                        # Random spatial jitter (+/- 25% of crop size)
                        jitter_y = random.randint(-self.crop_h // 4, self.crop_h // 4)
                        jitter_x = random.randint(-self.crop_w // 4, self.crop_w // 4)

                        top = int(np.clip(cy - self.crop_h // 2 + jitter_y, 0, max(0, h - self.crop_h)))
                        left = int(np.clip(cx - self.crop_w // 2 + jitter_x, 0, max(0, w - self.crop_w)))
                        bottom = top + self.crop_h
                        right = left + self.crop_w

                        crop_img = img.crop((left, top, right, bottom))
                        crop_mask = mask.crop((left, top, right, bottom))

                        img_tensor, mask_tensor = self.post_crop_transforms(crop_img, crop_mask)

                        if self.return_metadata:
                            meta = {
                                "sequence": item["sequence"],
                                "stem": item["stem"],
                                "rgb_path": rgb_path,
                                "gt_path": gt_path,
                                "rcs_class": c,
                            }
                            return img_tensor, mask_tensor, meta
                        return img_tensor, mask_tensor
                except Exception:
                    pass  # Fall back to standard sample if any I/O error

        # Standard deterministic / random sample
        item = self.samples[idx]
        rgb_path = item["rgb_path"]
        gt_path = item["gt_path"]

        with Image.open(rgb_path) as img:
            img = img.convert("RGB")

        mask = None
        if gt_path is not None:
            with Image.open(gt_path) as m:
                mask = m.copy()

        if self.transforms is not None:
            img_tensor, mask_tensor = self.transforms(img, mask)
        else:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(np.array(mask)).long() if mask is not None else None

        if self.return_metadata:
            meta = {
                "sequence": item["sequence"],
                "stem": item["stem"],
                "rgb_path": rgb_path,
                "gt_path": gt_path,
            }
            if mask_tensor is not None:
                return img_tensor, mask_tensor, meta
            return img_tensor, meta

        if mask_tensor is not None:
            return img_tensor, mask_tensor
        return img_tensor


def build_dataloader(
    split: str = "train",
    data_root: str = "dataset",
    splits_dir: str = "splits",
    batch_size: int = 2,
    num_workers: int = 0,
    crop_size: Tuple[int, int] = (512, 512),
    eval_size: Tuple[int, int] = (512, 1024),
    shuffle: Optional[bool] = None,
    rcs_enabled: bool = False,
    rcs_prob: float = 0.5,
    rcs_stats_file: Optional[str] = "data/processed/rcs_training_stats.json",
) -> DataLoader:
    """Build and return DataLoader for train or test split.

    Args:
        split: 'train' or 'test'.
        data_root: Dataset root directory.
        splits_dir: Directory containing train.txt and test.txt.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.
        crop_size: (H, W) random crop for training.
        eval_size: (H, W) resize for testing.
        shuffle: Whether to shuffle data (defaults to True for train, False for test).
        rcs_enabled: Enable Rare-Class Crop Sampling for training.
        rcs_prob: Probability of rare-class crops.
        rcs_stats_file: Path to RCS stats JSON.

    Returns:
        torch.utils.data.DataLoader
    """
    is_train = split.lower() == "train"
    split_filename = "train.txt" if is_train else "test.txt"
    split_file = Path(splits_dir) / split_filename

    if is_train:
        transforms = get_train_transforms(crop_size=crop_size)
        should_shuffle = True if shuffle is None else shuffle
    else:
        transforms = get_test_transforms(eval_size=eval_size)
        should_shuffle = False if shuffle is None else shuffle

    dataset = ACDCDataset(
        split_file=str(split_file),
        data_root=data_root,
        transforms=transforms,
        return_metadata=True,
        rcs_enabled=(rcs_enabled and is_train),
        rcs_prob=rcs_prob,
        rcs_stats_file=rcs_stats_file,
        crop_size=crop_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )
    return dataloader
