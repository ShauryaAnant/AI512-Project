# ACDC Night Dataset Audit Report

**Date of Audit:** 2026-11-05  
**Audited Directory Root:** `C:\Users\shaur\AI512 project\dataset`  
**Audit Script:** `scripts/audit_dataset.py`

---

## 1. Split & Sequence Image Counts

| Split | Images Verified | Sequences (Images) | Ground Truth Present? |
|---|---|---|---|
| **`train`** | **400** | `GOPR0351` (93), `GOPR0376` (147), `GP010376` (56), `GP010397` (60), `GP020397` (44) | **Yes** (`gt/gt/night/train`) |
| **`val`** | **106** | `GOPR0351` (56), `GOPR0356` (50) | **Yes** (`gt/gt/night/val`) |
| **`test`** | **500** | `GOPR0355` (10), `GOPR0356` (179), `GOPR0364` (105), `GOPR0594` (148), `GP010364` (46), `GP010594` (12) | **No (Withheld for Benchmark Server)** |
| **`train_ref`** | **400** | Same 5 sequences (GPS-matched day images) | No labels |
| **`val_ref`** | **106** | Same 2 sequences (GPS-matched day images) | No labels |
| **`test_ref`** | **500** | Same 6 sequences (GPS-matched day images) | No labels |

> [!NOTE]
> All image counts match the official ACDC paper (Sakaridis et al., ICCV 2021) exactly: **400 train, 106 val, 500 test**.

---

## 2. Image Dimensions & Encoding
- **Spatial Resolution:** All 2012 images in the night split are exactly **1920 × 1080** (GoPro 1080p).
- **Color Mode:** 8-bit RGB (`RGB`).
- **Format:** Lossless PNG.

---

## 3. Ground Truth & Annotation Verification
- **`*_gt_labelTrainIds.png`:**
  - Mode: `L` (8-bit grayscale).
  - Unique Values in Dataset: `[np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(3), np.uint8(4), np.uint8(5), np.uint8(6), np.uint8(7), np.uint8(8), np.uint8(9), np.uint8(10), np.uint8(11), np.uint8(12), np.uint8(13), np.uint8(14), np.uint8(15), np.uint8(16), np.uint8(17), np.uint8(18), np.uint8(255)]`.
  - Value Set: Strictly subset of [0..18, 255]. All 19 valid semantic classes (0..18) and void/ignore index (255) are verified present.
  - **Zero invalid or out-of-range labels detected.**
- **`*_gt_labelIds.png`:**
  - Raw Cityscapes IDs format: values in `[np.uint8(0), np.uint8(7), np.uint8(8), np.uint8(11), np.uint8(12), np.uint8(13), np.uint8(17), np.uint8(19), np.uint8(20), np.uint8(21), np.uint8(22), np.uint8(23), np.uint8(24), np.uint8(25), np.uint8(26), np.uint8(27), np.uint8(28), np.uint8(31), np.uint8(32), np.uint8(33)]` ($0\dots33$).
  - **Rule Check:** Confirmed that raw IDs are present but will **never** be used for training or evaluation.
- **`*_gt_invIds.png` (Invalid Regions Binary Mask):**
  - Mode: `L`.
  - Unique Values: `[np.uint8(0), np.uint8(1)]` where `0 = valid`, `1 = invalid` (regions with ambiguous semantic content).
- **`*_gt_invGray.png` (Invalid Regions Visualization Mask):**
  - Mode: `L`.
  - Unique Values: `[np.uint8(0), np.uint8(255)]` where `0 = valid`, `255 = invalid` (for visualization).

---

## 4. Pairing & Integrity Check
- **RGB to Ground Truth Pairing:** Exactly 1-to-1 match. Unmatched files: **0**.
- **Adverse to Daytime Reference Pairing:** Exactly 1-to-1 match across all splits. Missing reference images: **0**.
- **Test Set Withholding:** Confirmed `dataset/gt/gt/night/test` does not exist. Ground truth is strictly withheld.

---

## 5. Sequence Overlap & Same-Drive Leakage Analysis

| Sequence | Split(s) Present | Image Counts | Leakage Risk & Protocol Handling |
|---|---|---|---|
| **`GOPR0351`** | **`train`** AND **`val`** | Train: 93, Val: 56 | **Severe Leakage Risk:** Consecutive frames from the same drive are present in both train and val (frames separated by as few as 12 frames). |
| **`GOPR0356`** | **`val`** AND **`test`** | Val: 50, Test: 179 | **Zero Train Overlap:** Unseen during training. Represents the true, honest out-of-domain generalization performance. |
| `GOPR0376`, `GP010376`, `GP010397`, `GP020397` | `train` only | 147, 56, 60, 44 | Training set only. |
| `GOPR0355`, `GOPR0364`, `GOPR0594`, `GP010364`, `GP010594` | `test` only | 10, 105, 148, 46, 12 | Test set only. |

### Evaluation Protocol Mandate:
Validation results must **always** be reported across three distinct views:
1. **Val Overall (106 images)**
2. **Val-Seen-Drive (`GOPR0351`, 56 images)**
3. **Val-Unseen-Drive (`GOPR0356`, 50 images)** — Primary indicator of generalization.

---

## 6. Pixel Statistics & Class Distribution

| Metric | Train Split | Val Split |
|---|---|---|
| Total Pixels | 829,440,000 | 219,801,600 |
| Ignore Pixels (`255`) | 85,521,766 (10.31%) | 17,366,374 (7.90%) |
| Invalid Mask Pixels (`invIds = 1`) | 281,900,932 (33.99%) | 86,335,677 (39.28%) |

### Per-Class Pixel Counts (Train vs Val):
| Class ID | Class Name | Train Pixels | % Train | Val Pixels | % Val |
|---|---|---|---|---|---|
| 0 | road | 251,770,125 | 30.35% | 53,593,801 | 24.38% |
| 1 | sidewalk | 61,088,259 | 7.37% | 17,431,927 | 7.93% |
| 2 | building | 147,157,443 | 17.74% | 38,090,122 | 17.33% |
| 3 | wall | 14,054,749 | 1.69% | 4,022,306 | 1.83% |
| 4 | fence | 10,223,985 | 1.23% | 5,122,854 | 2.33% |
| 5 | pole | 7,845,730 | 0.95% | 1,600,887 | 0.73% |
| 6 | traffic light | 1,751,850 | 0.21% | 172,791 | 0.08% |
| 7 | traffic sign | 3,100,472 | 0.37% | 774,115 | 0.35% |
| 8 | vegetation | 94,131,506 | 11.35% | 38,089,324 | 17.33% |
| 9 | terrain | 5,435,223 | 0.66% | 2,197,247 | 1.00% |
| 10 | sky | 119,175,815 | 14.37% | 36,795,621 | 16.74% |
| 11 | person | 1,227,313 | 0.15% | 350,880 | 0.16% |
| 12 | rider | 325,000 | 0.04% | 21,587 | 0.01% |
| 13 | car | 17,716,814 | 2.14% | 1,141,109 | 0.52% |
| 14 | truck | 177,153 | 0.02% | 31,471 | 0.01% |
| 15 | bus | 1,746,314 | 0.21% | 0 | 0.00% |
| 16 | train | 5,993,415 | 0.72% | 2,457,411 | 1.12% |
| 17 | motorcycle | 345,828 | 0.04% | 82,299 | 0.04% |
| 18 | bicycle | 651,240 | 0.08% | 459,474 | 0.21% |

---

## 7. Illumination Statistics Summary (from `data/processed/illum_stats.csv`)
- **Mean Luma:** Train = 48.21 ± 12.85, Val = 41.60 ± 11.01
- **Standard Deviation of Luma (Contrast):** Train = 37.58, Val = 35.12
- **Percentage Very Dark Pixels (<10 luma):** Train = 15.29%, Val = 19.50%
- **Percentage Glare/Saturated Pixels (≥250 luma):** Train = 0.40%, Val = 0.26%

---

## 8. License & Usage Compliance
- **Source:** ETH Zurich Computer Vision Lab (ACDC: The Adverse Conditions Dataset with Correspondences).
- **License File:** `dataset/gt/License.pdf` and `dataset/rgb_anon/License.pdf` verified.
- **Compliance Rules:**
  1. Non-commercial research use only.
  2. Dataset images, annotations, and visual predictions overlaid on dataset images **must not be committed to GitHub or publicly redistributed**.
  3. All training and evaluation runs uploaded to Kaggle must remain in **Private Datasets**.
  4. Proper citation to Sakaridis et al., ICCV 2021 must be maintained in the final report.
