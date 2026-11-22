# Baseline Reproduction Results (ACDC Night)

**Architecture:** Supplied Mamba U-Net (`src/models/model.py`)  
**Hyperparameters:** 300 epochs, initial lr = 0.01, AdamW optimizer, CosineAnnealingLR  
**Loss:** Combined Multi-Class Cross-Entropy (`ignore_index=255`) + Soft-Dice Loss  
**Hardware:** Kaggle NVIDIA Tesla T4 GPU (16GB)  
**Dataset Split:** Supervised training on 400 ACDC Night images; evaluated on 106 test images  
**Batch Size:** 4 (with `drop_last=True`, yielding 100 training batches per epoch = all 400 images used)  
**Training Crop:** 512 × 512 random crop with scale jitter [0.75, 1.25]  
**Evaluation Resolution:** 512 × 1024 (resized from native 1920 × 1080)  

---

## 1. Overall Baseline Benchmark Table

| Run ID / Seed | Epochs | Train mIoU Best (%) | Test mIoU (%) | Dice Mean (%) | Pixel Accuracy (%) | mPA / Sensitivity (%) | Specificity Macro (%) | Params (M) | GFLOPs (1080p) |
|---|---|---|---|---|---|---|---|---|---|
| **Seed 42** | 300 | 23.51% (at 512×512) | **20.70%** (at 512×1024) | 27.16% | 73.70% | 36.49% | 98.38% | 0.0270 | 1.02 |

> **Note on mIoU discrepancy:** During training, the validation loop evaluates at the same 512 × 512 crop resolution used for training, yielding a best mIoU of 23.51%. The final test-time inference in `inference.py` evaluates at 512 × 1024 (wider aspect ratio), which introduces a resolution mismatch that lowers the score to 20.70%.

---

## 2. Per-Class IoU Breakdown (%) — Seed 42

| Class ID | Class Name | IoU (%) | Dice (%) | Category | Performance Assessment |
|---|---|---|---|---|---|
| 0 | road | **79.70%** | 88.70% | Flat / Surface | High (dominant ground plane) |
| 1 | sidewalk | **40.75%** | 57.90% | Flat / Surface | Moderate |
| 2 | building | **61.50%** | 76.16% | Construction | Strong spatial structure |
| 3 | wall | **16.08%** | 27.70% | Construction | Low (confused with building/fence) |
| 4 | fence | **4.28%** | 8.21% | Construction | Severe degradation |
| 5 | pole | **32.46%** | 49.02% | Object | Moderate (thin vertical geometry) |
| 6 | traffic light | **20.58%** | 34.14% | Object | Low (glare / saturated blooming) |
| 7 | traffic sign | **5.00%** | 9.53% | Object | Severe degradation |
| 8 | vegetation | **56.32%** | 72.06% | Nature | Moderate (dark foliage textures) |
| 9 | terrain | **4.70%** | 8.98% | Nature | Severe degradation |
| 10 | sky | **71.83%** | 83.60% | Sky | Strong (dark uniform upper half) |
| 11 | person | **0.05%** | 0.11% | Human | Catastrophic failure |
| 12 | rider | **0.00%** | 0.00% | Human | Complete failure |
| 13 | car | **0.00%** | 0.00% | Vehicle | Complete failure |
| 14 | truck | **0.00%** | 0.00% | Vehicle | Complete failure |
| 15 | bus | **N/A** (0 GT px) | N/A | Vehicle | Safely excluded (0 test pixels) |
| 16 | train | **0.00%** | 0.00% | Vehicle | Complete failure |
| 17 | motorcycle | **0.00%** | 0.00% | Vehicle | Complete failure |
| 18 | bicycle | **0.00%** | 0.00% | Vehicle | Complete failure |

---

## 3. Key Observations & Failure Hypotheses

1. **Catastrophic Vehicle & Human Collapse:** All dynamic objects (cars, persons, riders, trucks, bikes) achieved essentially 0% IoU. Under low-light conditions and without pre-training on large daytime datasets, the lightweight model failed to detect moving foreground objects against dark asphalt/backgrounds.
2. **Dominance of Large Spatial Structures:** Almost all positive mIoU is driven by static background classes: *road* (79.7%), *sky* (71.8%), *building* (61.5%), and *vegetation* (56.3%).
3. **Thin & Small Object Vulnerability:** Small traffic objects (*traffic signs* 5.0%, *fences* 4.3%, *traffic lights* 20.6%) suffer heavily from spatial resolution loss and glare blooming.
