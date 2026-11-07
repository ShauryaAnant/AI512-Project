# Kaggle Execution Guide — Phase 7: Baseline Reproduction

Follow these step-by-step instructions to train the baseline model on a free Kaggle GPU.

---

## Step 1: Upload ACDC Night as a Private Kaggle Dataset

1. On Kaggle, click **Create** > **New Dataset**.
2. Upload the `dataset` folder (specifically the `gt/gt/night` and `rgb_anon/rgb_anon/night` folders, preserving the double-nested structure).
3. Name your dataset (e.g., `acdc-night`).
4. **IMPORTANT (Licensing):** Set visibility to **PRIVATE** (do not make it public per ACDC license).
5. Once uploaded, note the mount path shown in the right sidebar (typically `/kaggle/input/acdc-night`).

---

## Step 2: Create and Configure a Kaggle Notebook

1. Click **Create** > **New Notebook**.
2. In the right-hand **Notebook Settings** panel:
   * **Accelerator:** Select **GPU T4** (or **GPU T4 x2**).
   * **Internet:** Toggle **ON** (required to clone your GitHub repo and install packages).
3. Under **Input Data** in the right sidebar, click **Add Data** and select your private `acdc-night` dataset.

---

## Step 3: Run the Notebook Cells

### Cell 1: Clone Repository & Install Dependencies
Replace `<YOUR_GITHUB_USERNAME>` and `<YOUR_REPO_NAME>` with your repository details:
```python
# Clone your code from GitHub
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git repo
%cd repo

# Install dependencies
!pip install -r requirements.txt
```

### Cell 2: Verify Setup & Check GPU
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Run model unit test to ensure everything works
!python tests/test_model.py
```

### Cell 3: Find Dataset Root Path
Verify the exact path where your dataset is mounted:
```python
import os
from pathlib import Path

# Find where dataset files are mounted
for p in Path("/kaggle/input").rglob("night"):
    print("Found night path at:", p)
```
*(Suppose your path is `/kaggle/input/acdc-night`, set that in the commands below).*

---

### Cell 4: Train Baseline (Seed 42)
Runs 300 epochs at $lr = 0.01$ with AdamW and CosineAnnealingLR. Automatically saves `last.pth` after each epoch and `best.pth` on test improvement.
```bash
!python train.py \
    --config configs/baseline_acdc_night.yaml \
    --data-root /kaggle/input/acdc-night \
    --out-dir /kaggle/working/baseline_seed42 \
    --seed 42 \
    --epochs 300 \
    --lr 0.01 \
    --batch-size 4
```

> [!TIP]
> **Resuming if disconnected:** If your Kaggle session reaches its time limit or disconnects, re-run the exact same command with `--resume` added to pick up from the last epoch instantly:
> `!python train.py --config configs/baseline_acdc_night.yaml --data-root /kaggle/input/acdc-night --out-dir /kaggle/working/baseline_seed42 --resume`

---

### Cell 5: Evaluate & Export Metrics (`results.json`)
Run inference on the 106 test images at full $1920 \times 1080$ resolution:
```bash
!python inference.py \
    --config configs/baseline_acdc_night.yaml \
    --checkpoint /kaggle/working/baseline_seed42/checkpoints/best.pth \
    --out-dir /kaggle/working/baseline_seed42/eval
```

---

### Cell 6: Repeat for Seeds 1337 & 2026 (Optional / Multi-Seed)
Per scientific protocol, repeat for seed 1337 and seed 2026:
```bash
# Seed 1337
!python train.py --config configs/baseline_acdc_night.yaml --data-root /kaggle/input/acdc-night --out-dir /kaggle/working/baseline_seed1337 --seed 1337 --epochs 300 --lr 0.01 --batch-size 4
!python inference.py --config configs/baseline_acdc_night.yaml --checkpoint /kaggle/working/baseline_seed1337/checkpoints/best.pth --out-dir /kaggle/working/baseline_seed1337/eval

# Seed 2026
!python train.py --config configs/baseline_acdc_night.yaml --data-root /kaggle/input/acdc-night --out-dir /kaggle/working/baseline_seed2026 --seed 2026 --epochs 300 --lr 0.01 --batch-size 4
!python inference.py --config configs/baseline_acdc_night.yaml --checkpoint /kaggle/working/baseline_seed2026/checkpoints/best.pth --out-dir /kaggle/working/baseline_seed2026/eval
```

---

### Cell 7: Download Output Results
Zip and download `results.json`, logs, and metric CSVs to your local machine:
```bash
!zip -r baseline_results.zip /kaggle/working/baseline_*/eval/results.json /kaggle/working/baseline_*/*.csv /kaggle/working/baseline_*/*.log
```
Then copy the `results.json` into your local `runs/baseline/` folder so we can generate the final baseline comparison tables in Phase 8!
