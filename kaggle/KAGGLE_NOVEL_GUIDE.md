# Kaggle Execution Guide — Phase 11 & 12: BOLT Novel Method Training

Follow these step-by-step instructions to train the novel **BOLT-Loss** model on your Kaggle GPU notebook.

---

## What is Being Trained:
- **Model Architecture:** Supplied Mamba U-Net (26,975 parameters / 0.0270M) — strictly identical to baseline.
- **Novel Loss:** BOLT-Loss (Class-Weighted Boundary Loss + Inverse-Frequency Rebalancing + OHEM).
- **Protocol:** 300 epochs, initial $lr = 0.01$, AdamW, CosineAnnealingLR, batch size 4 (Seed 42).
- **Inference Overhead:** **0 extra parameters, 0 extra FLOPs.**

---

## Step 1: Open Your Existing Kaggle Notebook
1. Open the same Kaggle notebook you used for the baseline run.
2. Ensure GPU T4 accelerator and Internet are turned **ON**.
3. Ensure your private dataset (`acdc-night`) is mounted at `/kaggle/input/datasets/shauryaanant/acdc-night` (or `/kaggle/input/acdc-night`).

---

## Step 2: Pull Latest Code from GitHub

Run this code cell in Kaggle:

```bash
# If you are in /kaggle/working/repo:
%cd /kaggle/working/repo
!git pull origin main

# Verify novel files exist:
!python -m unittest tests.test_losses
```

*(If you start a fresh notebook, run the usual clone cell)*:
```bash
%cd /kaggle/working
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git repo
%cd repo
!pip install -r requirements.txt
```

---

## Step 3: Run BOLT-Loss Training (Seed 42)

Run this cell to start the 300-epoch training:

```bash
!python train.py \
    --config configs/novel_bolt_acdc_night.yaml \
    --data-root /kaggle/input/datasets/shauryaanant/acdc-night \
    --out-dir /kaggle/working/novel_bolt_seed42 \
    --seed 42 \
    --epochs 300 \
    --lr 0.01 \
    --batch-size 4
```

> [!TIP]
> **Resuming if disconnected:** If your session disconnects or reaches the 12-hour limit, add `--resume` to pick up from the last epoch instantly:
> `!python train.py --config configs/novel_bolt_acdc_night.yaml --data-root /kaggle/input/datasets/shauryaanant/acdc-night --out-dir /kaggle/working/novel_bolt_seed42 --resume`

---

## Step 4: Run Inference & Generate `results.json`

Once training finishes, evaluate all 106 test images at full resolution:

```bash
!python inference.py \
    --config configs/novel_bolt_acdc_night.yaml \
    --checkpoint /kaggle/working/novel_bolt_seed42/checkpoints/best.pth \
    --data-root /kaggle/input/datasets/shauryaanant/acdc-night \
    --out-dir /kaggle/working/novel_bolt_seed42/eval
```

---

## Step 5: Bundle & Download Results

```python
import os
from IPython.display import FileLink

os.chdir('/kaggle/working')
!zip -r /kaggle/working/novel_bolt_seed42_bundle.zip novel_bolt_seed42/

FileLink('novel_bolt_seed42_bundle.zip')
```

Or download directly from Kaggle's right-side **Output** pane.
