"""Deterministic Split File Generator for ACDC Night Dataset.

Generates split text files in splits/ for training, validation,
seen/unseen sequence splits, and test sets.
"""

import argparse
from pathlib import Path
import sys


def make_splits(data_root_str: str = "dataset", out_dir_str: str = "splits"):
    data_root = Path(data_root_str).resolve()
    out_dir = Path(out_dir_str).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_night = data_root / "rgb_anon" / "rgb_anon" / "night"
    gt_night = data_root / "gt" / "gt" / "night"

    if not rgb_night.exists():
        print(f"Error: Path {rgb_night} does not exist.")
        sys.exit(1)

    print(f"Generating splits from: {data_root}")
    print(f"Saving splits to: {out_dir}")

    # 1. Train split (400 images)
    train_lines = []
    train_rgb_dir = rgb_night / "train"
    train_gt_dir = gt_night / "train"

    for seq_dir in sorted(train_rgb_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        for img_file in sorted(seq_dir.glob("*_rgb_anon.png")):
            stem = img_file.stem.replace("_rgb_anon", "")
            gt_file = train_gt_dir / seq_name / f"{stem}_gt_labelTrainIds.png"
            inv_file = train_gt_dir / seq_name / f"{stem}_gt_invIds.png"
            if not gt_file.exists():
                print(f"Missing GT: {gt_file}")
                continue

            rel_rgb = img_file.relative_to(data_root).as_posix()
            rel_gt = gt_file.relative_to(data_root).as_posix()
            rel_inv = inv_file.relative_to(data_root).as_posix() if inv_file.exists() else ""
            train_lines.append(f"{rel_rgb}\t{rel_gt}\t{rel_inv}\t{seq_name}\t{stem}")

    with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    print(f"Wrote {len(train_lines)} entries to train.txt (Training set: 400 images)")

    # 2. Test split (106 images from the labeled val folder)
    test_lines = []
    test_rgb_dir = rgb_night / "val"
    test_gt_dir = gt_night / "val"

    for seq_dir in sorted(test_rgb_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        for img_file in sorted(seq_dir.glob("*_rgb_anon.png")):
            stem = img_file.stem.replace("_rgb_anon", "")
            gt_file = test_gt_dir / seq_name / f"{stem}_gt_labelTrainIds.png"
            inv_file = test_gt_dir / seq_name / f"{stem}_gt_invIds.png"
            if not gt_file.exists():
                print(f"Missing GT: {gt_file}")
                continue

            rel_rgb = img_file.relative_to(data_root).as_posix()
            rel_gt = gt_file.relative_to(data_root).as_posix()
            rel_inv = inv_file.relative_to(data_root).as_posix() if inv_file.exists() else ""
            test_lines.append(f"{rel_rgb}\t{rel_gt}\t{rel_inv}\t{seq_name}\t{stem}")

    with open(out_dir / "test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")
    print(f"Wrote {len(test_lines)} entries to test.txt (Test set: 106 images)")

    assert len(train_lines) == 400, f"Expected 400 train, got {len(train_lines)}"
    assert len(test_lines) == 106, f"Expected 106 test, got {len(test_lines)}"
    print("Clean train (400) and test (106) splits generated and verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACDC Night split lists")
    parser.add_argument("--data-root", type=str, default="dataset", help="Path to ACDC dataset root")
    parser.add_argument("--out-dir", type=str, default="splits", help="Output directory for split files")
    args = parser.parse_args()
    make_splits(args.data_root, args.out_dir)
