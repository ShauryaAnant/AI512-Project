"""Step 0: Build Rare-Class Sampling (RCS) index and compute training set class statistics.

Scans the 400 ACDC Night training ground truth masks to:
1. Compute exact per-class pixel counts (f_c) and frequencies.
2. Count images containing at least MIN_PIXELS of class c (n_c).
3. Compute DAFormer-style inverse-square-root sampling probabilities P(c) ~ f_c^(-0.5).
4. Save class-to-images index for instant lookup in the dataset loader.
5. Compute mild class-balanced weights for Cross-Entropy: w_c ~ 1 / sqrt(f_c).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
from src.metrics import CLASS_NAMES

MIN_PIXELS_THRESHOLD = 50  # Ignore 1-pixel annotation noise
NUM_CLASSES = 19
IGNORE_INDEX = 255


def build_rcs_index(
    data_root: str = "dataset",
    train_split: str = "splits/train.txt",
    out_dir: str = "data/processed",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)

    train_lines = [line.strip().split("\t") for line in open(train_split) if line.strip()]
    print(f"Scanning {len(train_lines)} training images for Rare-Class Sampling...")

    class_pixel_counts = defaultdict(int)
    class_to_indices = defaultdict(list)
    class_image_counts = defaultdict(int)

    for idx, cols in enumerate(tqdm(train_lines, desc="Auditing train masks")):
        rgb_rel, gt_rel, inv_rel, seq, stem = cols
        gt_path = data_root / gt_rel

        mask = np.array(Image.open(gt_path), dtype=np.int32)
        valid = (mask != IGNORE_INDEX)

        # Count pixels per class
        for c in range(NUM_CLASSES):
            px_count = int(np.sum((mask == c) & valid))
            if px_count > 0:
                class_pixel_counts[c] += px_count
            if px_count >= MIN_PIXELS_THRESHOLD:
                class_to_indices[c].append(idx)
                class_image_counts[c] += 1

    total_valid_pixels = sum(class_pixel_counts.values())

    print("\n=== Training Split Class Statistics (N = 400 images) ===")
    print(f"{'ID':<4} {'Class Name':<15} {'Pixel Count':<12} {'Pixel Share (%)':<16} {'Image Count (n_c)':<18}")
    print("-" * 70)

    class_stats = {}
    f_c = np.zeros(NUM_CLASSES, dtype=np.float64)

    for c in range(NUM_CLASSES):
        px = class_pixel_counts[c]
        share = (px / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0.0
        n_c = class_image_counts[c]
        f_c[c] = px / total_valid_pixels if total_valid_pixels > 0 else 1e-8
        name = CLASS_NAMES[c]
        print(f"{c:<4} {name:<15} {px:<12} {share:<16.4f}% {n_c:<18}")

        class_stats[c] = {
            "name": name,
            "pixel_count": px,
            "pixel_share": share / 100.0,
            "image_count": n_c,
        }

    # Step 1: DAFormer-style Sampling Probability P(c) ~ f_c^(-0.5)
    # Exclude classes with 0 training images (if any)
    p_unnorm = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        if class_image_counts[c] > 0:
            p_unnorm[c] = (f_c[c] + 1e-8) ** -0.5
        else:
            p_unnorm[c] = 0.0

    p_sampling = p_unnorm / np.sum(p_unnorm)

    print("\n=== Rare-Class Sampling Probabilities P(c) ~ f_c^(-0.5) ===")
    for c in range(NUM_CLASSES):
        print(f"{c:<4} {CLASS_NAMES[c]:<15}: {p_sampling[c]*100:.2f}% (Images available: {class_image_counts[c]})")

    # Step 2: Mild Class-Balanced CE Weights: w_c ~ 1 / sqrt(f_c)
    # Clipped at [0.2, 5.0], renormalized to mean = 1.0
    w_raw = (f_c + 1e-8) ** -0.5
    # Normalize by median or mean
    w_raw = w_raw / np.mean(w_raw)
    # Clip to [0.2, 5.0] to prevent gradient instability on majority classes
    w_clipped = np.clip(w_raw, 0.2, 5.0)
    # Renormalize so average scale is exactly 1.0
    w_mild = NUM_CLASSES * (w_clipped / np.sum(w_clipped))

    print("\n=== Mild Class-Balanced CE Weights (Clipped at 5.0, Mean = 1.0) ===")
    for c in range(NUM_CLASSES):
        print(f"{c:<4} {CLASS_NAMES[c]:<15}: {w_mild[c]:.4f}")

    # Save to JSON files
    output_package = {
        "class_stats": class_stats,
        "sampling_probabilities": p_sampling.tolist(),
        "mild_class_weights": w_mild.tolist(),
        "class_to_indices": {int(k): v for k, v in class_to_indices.items()},
    }

    out_file = out_dir / "rcs_training_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_package, f, indent=2)
    print(f"\nSaved RCS training statistics & image index to: {out_file}")


if __name__ == "__main__":
    build_rcs_index()
