"""ACDC Night PyTorch Dataset and DataLoader constructor.

Loads RGB images and Cityscapes trainId annotations (0..18, 255 ignore index)
from generated split files.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.datasets.transforms import get_test_transforms, get_train_transforms


class ACDCDataset(Dataset):
    """PyTorch Dataset for ACDC Night semantic segmentation."""

    def __init__(
        self,
        split_file: str,
        data_root: str = "dataset",
        transforms: Optional[Callable] = None,
        return_metadata: bool = True,
    ) -> None:
        """Initialize ACDCDataset.

        Args:
            split_file: Path to split text file (e.g. splits/train.txt or splits/test.txt).
            data_root: Root directory of ACDC dataset.
            transforms: Callable joint transform applied to (image, mask).
            return_metadata: If True, returns dict with sequence name and image stem.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.transforms = transforms
        self.return_metadata = return_metadata

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
        item = self.samples[idx]
        rgb_path = item["rgb_path"]
        gt_path = item["gt_path"]

        # Load RGB image
        with Image.open(rgb_path) as img:
            img = img.convert("RGB")

        # Load GT mask (8-bit trainIds) if available
        mask = None
        if gt_path is not None:
            with Image.open(gt_path) as m:
                mask = m.copy()

        # Apply transforms
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
