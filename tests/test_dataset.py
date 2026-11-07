"""Unit tests for ACDCDataset, transforms, and DataLoader."""

import torch
from src.datasets.acdc import ACDCDataset, build_dataloader
from src.datasets.transforms import get_test_transforms, get_train_transforms


def test_dataset_item():
    """Verify loading single item from ACDCDataset."""
    ds = ACDCDataset(
        split_file="splits/train.txt",
        data_root="dataset",
        transforms=get_train_transforms(crop_size=(256, 256)),
        return_metadata=True,
    )
    assert len(ds) == 400, f"Expected 400 train items, got {len(ds)}"

    img, mask, meta = ds[0]
    assert img.shape == (3, 256, 256), f"Expected (3, 256, 256), got {img.shape}"
    assert mask.shape == (256, 256), f"Expected (256, 256), got {mask.shape}"
    assert mask.dtype == torch.long, f"Expected torch.long, got {mask.dtype}"

    # Check unique label values
    unique_vals = set(mask.unique().tolist())
    for val in unique_vals:
        assert (0 <= val <= 18) or (val == 255), f"Unexpected label value {val}"

    assert "sequence" in meta and "stem" in meta
    print("test_dataset_item passed!")


def test_eval_transforms():
    """Verify evaluation transforms produce exact resolution (divisible by 32)."""
    ds = ACDCDataset(
        split_file="splits/test.txt",
        data_root="dataset",
        transforms=get_test_transforms(eval_size=(512, 1024)),
        return_metadata=True,
    )
    assert len(ds) == 106, f"Expected 106 test items, got {len(ds)}"

    img, mask, meta = ds[0]
    assert img.shape == (3, 512, 1024), f"Expected (3, 512, 1024), got {img.shape}"
    assert mask.shape == (512, 1024), f"Expected (512, 1024), got {mask.shape}"
    assert 512 % 32 == 0 and 1024 % 32 == 0, "Dimensions must be divisible by 32"
    print("test_eval_transforms passed!")


def test_dataloader_batch():
    """Verify DataLoader batch iteration."""
    loader = build_dataloader(
        split="train",
        data_root="dataset",
        splits_dir="splits",
        batch_size=2,
        num_workers=0,
        crop_size=(128, 256),
    )

    batch = next(iter(loader))
    imgs, masks, metas = batch

    assert imgs.shape == (2, 3, 128, 256), f"Expected (2, 3, 128, 256), got {imgs.shape}"
    assert masks.shape == (2, 128, 256), f"Expected (2, 128, 256), got {masks.shape}"
    assert len(metas["stem"]) == 2
    print("test_dataloader_batch passed!")


if __name__ == "__main__":
    test_dataset_item()
    test_eval_transforms()
    test_dataloader_batch()
    print("All Phase 4 dataset and DataLoader tests passed successfully!")
