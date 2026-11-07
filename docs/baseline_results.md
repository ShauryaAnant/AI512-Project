# Baseline Reproduction Results (ACDC Night)

**Architecture:** Supplied Mamba U-Net (`src/models/model.py`)  
**Hyperparameters:** 300 epochs, initial $lr = 0.01$, AdamW optimizer, CosineAnnealingLR  
**Loss:** Combined Multi-Class Cross-Entropy (`ignore_index=255`) + Soft-Dice Loss  
**Hardware:** Kaggle NVIDIA T4 GPU  
**Dataset Split:** Supervised training on 400 ACDC Night images; evaluated on 106 test images  

---

## 1. Overall Baseline Benchmark Table

| Run ID / Seed | Epochs | Test mIoU (%) | Dice Mean (%) | Pixel Accuracy (%) | Sensitivity Macro (%) | Specificity Macro (%) | Params (M) | GFLOPs | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| **Seed 42** | 300 | TBD | TBD | TBD | TBD | TBD | 0.0270 | 0.13 | TBD | TBD |
| **Seed 1337** | 300 | TBD | TBD | TBD | TBD | TBD | 0.0270 | 0.13 | TBD | TBD |
| **Seed 2026** | 300 | TBD | TBD | TBD | TBD | TBD | 0.0270 | 0.13 | TBD | TBD |
| **Mean ± Std** | 300 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | 0.0270 | 0.13 | TBD | TBD |

---

## 2. Per-Class IoU Breakdown (%)

| Class ID | Class Name | Seed 42 | Seed 1337 | Seed 2026 | Mean ± Std |
|---|---|---|---|---|---|
| 0 | road | TBD | TBD | TBD | TBD |
| 1 | sidewalk | TBD | TBD | TBD | TBD |
| 2 | building | TBD | TBD | TBD | TBD |
| 3 | wall | TBD | TBD | TBD | TBD |
| 4 | fence | TBD | TBD | TBD | TBD |
| 5 | pole | TBD | TBD | TBD | TBD |
| 6 | traffic light | TBD | TBD | TBD | TBD |
| 7 | traffic sign | TBD | TBD | TBD | TBD |
| 8 | vegetation | TBD | TBD | TBD | TBD |
| 9 | terrain | TBD | TBD | TBD | TBD |
| 10 | sky | TBD | TBD | TBD | TBD |
| 11 | person | TBD | TBD | TBD | TBD |
| 12 | rider | TBD | TBD | TBD | TBD |
| 13 | car | TBD | TBD | TBD | TBD |
| 14 | truck | TBD | TBD | TBD | TBD |
| 15 | bus (0 GT pixels) | N/A | N/A | N/A | N/A |
| 16 | train | TBD | TBD | TBD | TBD |
| 17 | motorcycle | TBD | TBD | TBD | TBD |
| 18 | bicycle | TBD | TBD | TBD | TBD |
