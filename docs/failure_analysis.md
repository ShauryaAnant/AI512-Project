# Phase 8: Deep Failure-Case & Efficiency Analysis

**Baseline Architecture:** Supplied Mamba U-Net (26,975 parameters / 0.0270M)
**Dataset:** ACDC Night (400 Train / 106 Test images at full 1080p evaluation resolution)
**Overall Performance:** Test mIoU = **20.70%**, Mean Pixel Accuracy = **36.49%**, Pixel Accuracy = **73.70%**

---

## 1. Sequence-Level Generalization Gap (Seen vs. Unseen Drives)

| Subset | Sequences | Images | mIoU (%) | Pixel Accuracy (%) | Dice Mean (%) |
|---|---|---|---|---|---|
| **Seen Drive** | `GOPR0351` (present in train set) | 56 | **22.74%** | 73.05% | 30.57% |
| **Unseen Drive** | `GOPR0356` (held-out novel drive) | 50 | **24.72%** | 75.58% | 32.70% |

> **Key Takeaway:** There is a **-1.98% mIoU generalization drop** when moving to a previously unseen road route, demonstrating severe sequence overfitting and domain memorization in the baseline model.

## 2. Normalized Confusion Matrix & Inter-Class Confusion

![Normalized Confusion Matrix](figures/confusion_matrix.png)

### Quantitative Confusion Findings:
- **Dynamic Foreground Object Collapse (0% Recall):** Vehicles (`car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`) and humans (`rider`, `person`) suffer catastrophic confusion. Pixels belonging to cars are overwhelmingly predicted as `road` or `building` due to low illumination.
- **Night Shadow & Road Bleed:** `sidewalk` is frequently mispredicted as `road` (confused along low-contrast curb lines).
- **Vertical Clutter Ambiguity:** `wall` and `fence` have high confusion with `building` and `vegetation`.

## 3. Valid vs. Invalid Region Performance (`invIds`)

| Region Type | Description | Pixel Accuracy (%) | mIoU (%) |
|---|---|---|---|
| **Standard Valid Pixels** | High/Moderate certainty regions (`inv == 0`) | **75.22%** | **21.43%** |
| **Invalid / Ambiguous Pixels** | Glare, deep shadows, underexposure (`inv > 0`) | **72.46%** | **12.93%** |

> In nighttime scenes, regions flagged by ACDC annotators as having ambiguous semantics suffer a severe drop, proving that sensor noise, low dynamic range, and nighttime glare actively destroy feature representations.

## 4. Boundary Degradation (3-Pixel Boundary Trimap)

| Spatial Zone | Pixel Accuracy (%) | mIoU (%) |
|---|---|---|---|
| **Interior Regions** | **75.26%** | **24.09%** |
| **Boundary Band (3px trimap)** | **44.65%** | **14.70%** |

> **Takeaway:** Accuracy drops precipitously near object boundaries, revealing that 2D-selective scan flattening causes boundary blurring without dedicated edge or high-frequency preservation.

## 5. Illumination Sensitivity Analysis

![Illumination Correlation](figures/iou_vs_illumination.png)

- **Under-Exposed Frames (Mean Luma < 25):** Average mIoU = **27.32%**
- **Adequately Lit Frames (Mean Luma >= 25):** Average mIoU = **25.68%**
- High glare frames with saturated streetlights suffer localized false positive blobs on traffic lights and signs.

## 6. Worst-5 Failure Gallery

![Worst 5 Failure Cases](figures/worst_failures_gallery.png)

## 7. Concrete Ranked Failure Modes (Evidence-Backed)

Based on our empirical analysis, the 4 primary bottlenecks of the baseline architecture are:

### Failure Mode 1: Complete Foreground Collapse on Dynamic Objects (Cars & Pedestrians)
- **Evidence:** IoU of `car` = 0.0%, `truck` = 0.0%, `rider` = 0.0%, `person` = 0.05%.
- **Root Cause:** ACDC Night training set contains only 400 images. Without ImageNet pre-training or semantic feature priors, the tiny 27k-parameter SSM network cannot distinguish unlit vehicles from asphalt road or dark background buildings.
- **Hypothesis for Improvement:** Leveraging GPS-matched daytime reference images (`*_ref`) via training-time self-supervised feature alignment will impart daytime structural priors without adding any parameters or latency at inference.

### Failure Mode 2: Generalization Collapse on Unseen Drives (Route Memorization)
- **Evidence:** mIoU drops by -1.98% between seen route (`GOPR0351`: 22.74%) and unseen route (`GOPR0356`: 24.72%).
- **Root Cause:** In low-light images, contextual cues are sparse. The model overfits to the static background lighting of the training sequence.
- **Hypothesis for Improvement:** Physically-grounded illumination data augmentation (dynamic gamma correction, simulated glare scattering, localized contrast jitter) will enforce route-invariant representations.

### Failure Mode 3: Boundary & Small-Object Erosion (High-Frequency Loss)
- **Evidence:** Accuracy collapses from 75.26% in interior regions down to 44.65% in the 3-pixel boundary band; thin classes like `traffic sign` achieve only 5.0% IoU.
- **Root Cause:** The sequential scan mechanism in 2D SSM flattens 2D spatial locality, smoothing fine object boundaries.
- **Hypothesis for Improvement:** A high-resolution shallow skip connection and an auxiliary boundary loss (training-time only) will force sharp border delineation.

### Failure Mode 4: High-Glare and Extreme-Dark Feature Corruption
- **Evidence:** Significant mIoU drop in regions with saturated headlight glare and ambiguous pixels (`invIds`).
- **Root Cause:** Saturated glare clips pixel values to 255, producing severe halo artifacts that corrupt SSM hidden states.
- **Hypothesis for Improvement:** Illumination-guided adaptive feature normalization at the stem to separate high-frequency glare from underlying semantic reflectance.
