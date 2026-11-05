# VM-UNet Reference Notes & Adaptation Requirements

## 1. Official VM-UNet Overview (JCruan519/VM-UNet)
The official VM-UNet repository was developed for binary/medical image segmentation (ISIC, Synapse):
- **Core Architecture:** Combines U-Net encoder-decoder structure with Visual State Space (VSS) blocks based on VMamba.
- **Optimizer:** AdamW (learning rate = 1e-3 default in VM-UNet, but overridden to lr = 0.01 per supervisor instructions).
- **Scheduler:** CosineAnnealingLR.
- **Loss Function:** `BceDiceLoss` = Binary Cross Entropy + Dice Loss.
- **Input Size:** Medical images resized typically to 256x256 or 512x512.
- **Pretrained Weights:** Pretrained VMamba-S on ImageNet-1K.

## 2. Adaptation Needed for 19-Class Cityscapes/ACDC Night Segmentation

| Component | Official VM-UNet (Medical) | Our Pipeline (ACDC Night) | Reason |
|---|---|---|---|
| **Number of Classes** | 1 (binary mask) | 19 classes | ACDC follows the Cityscapes 19-class semantic benchmark. |
| **Output Head** | 1 output channel + `sigmoid` | 19 output channels, **raw logits** | Multi-class cross-entropy expects unnormalized logits `(B, 19, H, W)`. |
| **Loss Function** | `BceDiceLoss` | Multi-class `CrossEntropyLoss(ignore_index=255)` + Multi-class `SoftDiceLoss(ignore_index=255)` | Non-labeled / void pixels have trainId 255 and must be excluded from loss calculation. |
| **Metrics** | Binary IoU, Dice coefficient | 19-class dataset-level confusion matrix: mIoU, per-class IoU, Dice, mPA, pixel accuracy, sensitivity/specificity macro | Cityscapes/ACDC standardized benchmark protocol. |
| **Input Resolutions** | 256x256 square | GoPro 1920x1080 native resized/cropped (e.g. 512x512 or 512x1024) | Divisible by 32 (5 stages of 2x downsampling). |
| **Validation Splits** | Standard k-fold / val split | Split into `val overall` (106), `val-seen-drive` (56), `val-unseen-drive` (50) | Accounts for sequence `GOPR0351` leakage between train and val. |
