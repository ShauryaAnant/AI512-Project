# Model Architecture & Shape Contract Notes

## 1. Overview
The supplied model (`src/models/model.py`, frozen baseline at `src/models/model_baseline.py`) is an ultra-lightweight U-Net architecture adapted from VM-UNet principles. Instead of heavy Vision Transformer or external compiled CUDA extensions, it uses a pure-PyTorch implementation of selective state space models (`CustomMamba`) with parallel prefix scans combined with multi-receptive field convolutions (`MCHLayer`).

## 2. Key Modules

### `CustomMamba`
- **Mechanism:** Implements linear time-invariant state-space recurrence with continuous-to-discrete discretization:
  $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
  $$y_t = C_t h_t + D x_t$$
- **Parallel Scan:** Uses Hillis-Steele prefix doubling in `_selective_scan_parallel` with $\lceil\log_2 L\rceil$ steps, avoiding sequential Python loops for sequence length $L = H \times W$.
- **Parameters:**
  - $d_{\text{model}}$: feature dimension
  - $d_{\text{state}} = 8$: SSM hidden state dimension
  - $d_{\text{conv}} = 2$: 1D causal depthwise convolution kernel size
  - $\Delta$ projected via low-rank parameterization ($d_{\text{model}} / 16$)

### `MCHLayer` (Multi-Channel Hybrid Layer)
Splits normalized input channels into 4 equal quarters:
1. **Branch 1 (Identity):** Direct pass-through of features.
2. **Branch 2 (Local Detail):** $3\times3$ Depthwise Convolution (`groups = C // 4`, padding 1).
3. **Branch 3 (Global Context):** `CustomMamba` SSM layer with learnable residual skip scale.
4. **Branch 4 (Extended Field):** $3\times3$ Dilated Depthwise Convolution (`dilation = 2`, padding 2).
The branches are concatenated along channels, normalized via LayerNorm, and linearly projected to `output_dim`.

### `Model` (U-Net Topology)
- **Encoder Stages:**
  - `encoder1`: Conv2d(3 -> 8) -> GroupNorm(4, 8) -> MaxPool(2) -> GELU -> $H/2, W/2$
  - `encoder2`: Conv2d(8 -> 16) -> GroupNorm(4, 16) -> MaxPool(2) -> GELU -> $H/4, W/4$
  - `encoder3`: Conv2d(16 -> 24) -> GroupNorm(4, 24) -> MaxPool(2) -> GELU -> $H/8, W/8$
  - `encoder4`: MCHLayer(24 -> 32) -> GroupNorm(4, 32) -> MaxPool(2) -> GELU -> $H/16, W/16$
  - `encoder5`: MCHLayer(32 -> 48) -> GroupNorm(4, 48) -> MaxPool(2) -> GELU -> $H/32, W/32$
  - `encoder6` (Bottleneck): MCHLayer(48 -> 64) -> GELU -> $H/32, W/32$
- **Decoder Stages:**
  - `decoder1`: MCHLayer(64 -> 48) + skip connection from `t5`
  - `decoder2`: MCHLayer(48 -> 32) + Bilinear Upsample(2x) + skip from `t4`
  - `decoder3`: MCHLayer(32 -> 24) + Bilinear Upsample(2x) + skip from `t3`
  - `decoder4`: Conv2d(24 -> 16) + Bilinear Upsample(2x) + skip from `t2`
  - `decoder5`: Conv2d(16 -> 8) + Bilinear Upsample(2x) + skip from `t1`
  - `final`: Conv2d(8 -> 19) + Bilinear Upsample(2x) -> $H, W$

## 3. Shape & Parameter Verification
- **Total Parameters:** 26,975 (~0.027M parameters)
- **Trainable Parameters:** 26,975
- **Input Constraints:** Spatial dimensions $(H, W)$ must be multiples of 32 due to 5 stages of $2\times$ pooling/upsampling.
- **Output Shape:** $(B, 19, H, W)$ raw unnormalized logits.
