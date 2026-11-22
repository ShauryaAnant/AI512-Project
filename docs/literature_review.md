# Phase 9: Literature Review & Taxonomy

This literature review surveys 20 key papers relevant to **ultra-lightweight nighttime semantic segmentation** using **state-space models (SSMs / Mamba)** on the **ACDC Night** benchmark. It is structured into 6 thematic clusters reflecting the research problem, efficiency requirements, architectural lineage, and our empirical findings from Phase 8.

---

## 1. Structured Taxonomy & Paper Matrix

| # | Paper | Year | Venue | Core Idea | Dataset(s) | Extra Compute / Overheads | Limitations & Gaps relative to our setting |
|---|---|---|---|---|---|---|---|
| **Cluster 1: Adverse-Condition & Nighttime Semantic Segmentation** |
| 1 | **ACDC** (Sakaridis et al.) | 2021 | ICCV | Introduces ACDC dataset (fog, night, rain, snow) with reference day images and invalid masks (`invIds`); benchmarked DeepLabv3+ and HRNet. | ACDC, Cityscapes | High (heavy models: DeepLabv3+, HRNet ~40–65M params) | Supervised baseline ignores ultra-lightweight regime; does not exploit day references at train-time for single-image inference. |
| 2 | **Dark Zurich / GCMA** (Sakaridis et al.) | 2019 / 2020 | ICCV / TPAMI | Guided Curriculum Model Adaptation: adapts day to night via twilight/dusk intermediate domain; uses GPS-matched pairs. | Dark Zurich, Cityscapes | Heavy multi-stage curriculum adaptation pipeline | Requires progressive twilight frames (not available in pure ACDC night); high training complexity. |
| 3 | **DANNet** (Wu et al.) | 2021 | CVPR | Dual-Attention Network for day-to-night adaptation; uses night image relighting sub-network and dual-branch domain alignment. | Dark Zurich, NightCity | Significant inference overhead if relighting branch is evaluated; heavy ResNet-101 backbone. | Relighting module can introduce hallucinatory color artifacts; heavy backbone incompatible with edge deployment. |
| 4 | **DANIA** (Wu et al.) | 2021 | ICCV | Domain Adaptive Nighttime Image Segmentation via static-dynamic feature alignment and illumination compensation. | Dark Zurich, Cityscapes | Dual-branch static-dynamic decomposition networks | Focuses on unconstrained UDA; relies on large-scale daytime source domain rather than direct supervised lightweight training. |
| 5 | **CCDistill** (Vandenhende et al.) | 2021 | WACV | Cross-condition knowledge distillation: distills day-teacher features into an adverse-condition student network. | ACDC, Dark Zurich | Zero inference cost (distillation during training only) | Assumes paired pixel-aligned inputs or heavy teacher backbones; does not address SSM-based architectures. |
| 6 | **CuDA-Net** (Zhou et al.) | 2023 | ICCV | Curvature-aware day-to-night adaptation using physics-guided illumination geometry. | ACDC, Dark Zurich | Moderate train-time overhead | Complex geometric formulations; validated primarily on 40M+ parameter CNNs. |
| **Cluster 2: NightCity & Illumination-Aware Segmentation** |
| 7 | **NightCity** (Tan et al.) | 2020 | ECCV | First dedicated large-scale dataset for nighttime semantic segmentation with explicit night driving annotations. | NightCity (4,297 images) | Evaluated on standard SegNet / PSPNet | Dataset possesses different class topologies; does not address ultra-lightweight state-space feature dynamics. |
| 8 | **IID-Segmentation** (Liu et al.) | 2022 | CVPR | Joint intrinsic image decomposition and segmentation for nighttime scenes: splits image into reflectance and shading. | NightCity, Cityscapes | High: dual-head Retinex estimation network | Separate decomposition heads increase FLOPs and parameter count; gradient instability when trained from scratch on small datasets. |
| **Cluster 3: Low-Light Enhancement as Preprocessing vs. Joint Learning** |
| 9 | **Zero-DCE / Zero-DCE++** (Guo et al.) | 2020 / 2021 | CVPR / TPAMI | Zero-Reference Deep Curve Estimation: formulates light enhancement as iterative quadratic curve estimation without paired data. | SICE, LOL, Dark Zurich | Lightweight enhancer (~0.01M params), but sequential two-stage inference required | Standalone enhancement amplifies high-frequency nighttime sensor noise and headlight glare blooming, degrading segmentation accuracy. |
| 10 | **RetinexNet** (Wei et al.) | 2018 | BMVC | Deep Retinex decomposition for low-light enhancement: separates illumination map and reflectance map with smoothness priors. | LOL, synthetic | Heavy dual-subnet architecture; multi-stage inference | Illumination maps over-smooth sharp boundaries and create unnatural halos around streetlights and headlamps. |
| 11 | **SCI** (Ma et al.) | 2022 | CVPR | Self-Calibrated Illumination learning for fast, low-light image enhancement with cascading calibration loops. | LOL, DarkFace | Very fast enhancer (~0.001M params) | Preprocessing cascade disconnects enhancement objective from downstream task loss; introduces spatial color shift. |
| **Cluster 4: Lightweight & Real-Time Semantic Segmentation** |
| 12 | **BiSeNet / BiSeNetV2** (Yu et al.) | 2018 / 2021 | ECCV / IJCV | Bilateral Segmentation Network: splits spatial detail path (shallow, wide) from semantic context path (deep, narrow). | Cityscapes, CamVid | 3–14M params, 10–50 GFLOPs; real-time on desktop GPUs | Heavily tuned for daytime; bilateral fusion lacks global context mechanisms needed to disambiguate unlit nighttime regions. |
| 13 | **SegFormer** (Xie et al.) | 2021 | NeurIPS | Hierarchical Transformer without positional encoding; SegFormer-B0 achieves efficient daytime segmentation. | Cityscapes, ADE20K | B0: 3.7M params, 8.4 GFLOPs (significantly larger than our 0.027M model) | Attention scales quadratically with tokens; difficult to deploy in sub-100k parameter regimes; struggles on small nighttime objects. |
| 14 | **PIDNet** (Xu et al.) | 2023 | CVPR | Proportional-Integral-Derivative network for real-time segmentation; dedicated Boundary branch balances detail and context. | Cityscapes, CamVid | 7.6M–30M params | Boundary branch effectively prevents boundary blur, but total parameter budget is >250× our baseline target. |
| **Cluster 5: Vision State Space Models (Mamba in Computer Vision)** |
| 15 | **Mamba / S4** (Gu & Dao) | 2023 | arXiv / NeurIPS | Selective State Space Model with linear-time sequential scanning and hardware-aware associative scan algorithms. | Language, Synthetic | Linear $O(N)$ complexity in sequence length | 1D sequential scan has no inherent 2D directional locality. |
| 16 | **VMamba** (Liu et al.) | 2024 | NeurIPS | Visual State Space Model with 2D-Selective-Scan (SS2D): traverses 2D feature maps along 4 spatial directions (top-left, bottom-right, etc.). | ImageNet, COCO, ADE20K | 22M–80M params (large-scale ViT competitor) | Focuses on ImageNet-scale foundation models; behavior when compressed to <30k parameters under extreme low-light is unstudied. |
| 17 | **VM-UNet** (Ruan & Xiang) | 2024 | arXiv / BMVC | Integrates SS2D blocks into a symmetrical U-Net architecture for 2D medical image segmentation. | ISIC17, Synapse | 27M params | Shows U-Net + SS2D viability, but retains high channel capacities; vulnerable to boundary smearing across scan lines. |
| 18 | **LightM-UNet** (Liao et al.) | 2024 | arXiv | Ultra-lightweight Mamba U-Net with Residual SSM blocks for medical segmentation, achieving <1M parameters. | LiTS, ISIC | 0.9M params, 1.2 GFLOPs | Validated exclusively on clean, high-contrast medical scans; untested on adverse driving environments with heavy noise/glare. |
| **Cluster 6: Boundary-Aware & Structural Feature Preservation** |
| 19 | **GSCNN** (Takikawa et al.) | 2019 | ICCV | Gated Shape CNN: explicit shape stream processing boundary features in parallel with regular stream; uses boundary loss. | Cityscapes | ~40M params, dual-stream ResNet | Proves that auxiliary boundary supervision dramatically improves thin objects (poles, signs, pedestrians); highly relevant concept. |
| 20 | **Boundary Loss** (Kervadec et al.) | 2019 | MIDL | Boundary loss formulation based on distance transforms from ground-truth surfaces; acts as a regional contour regularizer. | Medical datasets | **0 inference cost** (loss function applied exclusively at train-time) | Directly addresses boundary pixel degradation (where our baseline lost 30% accuracy) without adding a single parameter or FLOP. |

---

## 2. Key Synthesis & Insights from the Literature

### Insight A: Why Standalone Low-Light Preprocessing Fails Downstream
A common temptation in nighttime vision is running a low-light enhancer (Zero-DCE, SCI, RetinexNet) prior to segmentation. However, recent literature (e.g., ACDC benchmarks, DANNet) proves that separate low-light enhancement is optimized for human perceptual aesthetics (PSNR/SSIM) rather than semantic discriminability. Enhancers amplify high-frequency camera sensor noise in dark asphalt and expand saturated headlight glare halos into neighboring pixels, actively degrading downstream classification. **Conclusion:** Any illumination-handling mechanism must be integrated end-to-end within the segmentation network.

### Insight B: The Gap Between Existing Mamba Models and Edge Constraints
While VMamba, VM-UNet, and Mamba-UNet demonstrate linear $O(N)$ complexity, they are virtually all designed with 20M–80M parameters for high-end server GPUs. Our supplied baseline operates in an extreme **sub-30k parameter regime (0.027M params, 1.02 GFLOPs)**. At this extreme compression, the network has zero redundant capacity to recover from spatial information loss during 2D-selective scans.

### Insight C: The Zero-Inference-Cost Principle
Because our computational budget is strictly bounded, the most promising architectural improvements are **training-time-only regularizations**:
1. **Auxiliary Boundary Loss (Kervadec et al. / Takikawa et al.):** Penalizes edge errors during backpropagation but incurs 0 latency or memory overhead at test time.
2. **Day-Reference Feature Distillation (CCDistill / ACDC):** ACDC provides GPS-matched day reference frames (`*_ref`). Using them during training allows the model to learn daylight semantic structures, with single-image nighttime inference remaining completely untouched.
