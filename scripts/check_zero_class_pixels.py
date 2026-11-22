import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from src.metrics import CLASS_NAMES

test_txt = [line.strip().split('\t') for line in open('splits/test.txt') if line.strip()]
preds_dir = Path('runs/baseline_seed42/eval/preds')

gt_counts = {c: 0 for c in range(19)}
pred_counts = {c: 0 for c in range(19)}

for rgb_rel, gt_rel, inv_rel, seq, stem in test_txt:
    gt = np.array(Image.open(Path('dataset') / gt_rel))
    pred = np.array(Image.open(preds_dir / f'{stem}.png'))
    if pred.shape != gt.shape:
        pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
    
    valid = (gt != 255)
    for c in range(19):
        gt_counts[c] += int(np.sum((gt == c) & valid))
        pred_counts[c] += int(np.sum((pred == c) & valid))

print(f"{'Class ID':<8} {'Class Name':<15} {'GT Pixels':<12} {'Pred Pixels':<12} {'Ratio (Pred/GT)':<16}")
print("-" * 65)
for c in range(19):
    ratio = pred_counts[c] / gt_counts[c] if gt_counts[c] > 0 else 0.0
    print(f"{c:<8} {CLASS_NAMES[c]:<15} {gt_counts[c]:<12} {pred_counts[c]:<12} {ratio:<16.4f}")
