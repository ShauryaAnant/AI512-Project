# Phase 10: Novel Method Design (Method Specification)

**Method Name:** **BOLT-Loss** (**B**oundary-aware **O**nline-hard-example **L**oss with Class-Rebalanced **T**uning for Lightweight SSM Segmentation)  
**Target Platform:** Ultra-Lightweight Vision SSM U-Net (26,975 parameters / 0.0270M)  
**Benchmark:** ACDC Night (400 Train / 106 Test images, 19 classes)  
**Strict Hyperparameter Compliance:** 300 Epochs, Initial $lr = 0.01$, AdamW Optimizer, CosineAnnealingLR, Batch Size = 4  
**Inference Cost:** **0 Extra Parameters, 0 Extra FLOPs, 0 Added Latency**  

---

## 1. Technical Motivation & Alignment with Phase 8 Evidence

Our Phase 8 failure analysis and dataset audit revealed three critical bottlenecks in the baseline model:

1. **Catastrophic Foreground Dynamic Object Collapse (Failure Mode 1):**  
   All 8 dynamic vehicle and human classes (`car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `person`, `rider`) scored **0.0% IoU** (with `person` at 0.05%). In the ACDC night training set, 4 background classes (`road`, `building`, `sky`, `vegetation`) comprise **73.81%** of all pixels, while all 8 foreground dynamic classes together make up **less than 2.8%**. Standard unweighted Cross-Entropy gradients are 99% dominated by easy background regions, causing the model to minimize loss by never predicting foreground objects.
2. **Severe Boundary Smearing (Failure Mode 3):**  
   Pixel accuracy plummets from **75.26% in interior regions** to **44.65% along the 3-pixel boundary band** (a 30.6% drop). The 2D-selective scan mechanism in Vision SSMs traverses flattened 1D sequences, losing fine 2D spatial locality at object edges.
3. **Overfitting to Easy Background Context (Failure Mode 2 & 4):**  
   The model achieves 73.70% pixel accuracy merely by memorizing background lighting. Gradients from well-classified road and sky pixels swamp the optimizer.

### Core Architectural Principle: Zero Inference Overhead
Rather than modifying the lightweight 26,975-parameter backbone with bulky modules (which would violate the edge compute constraint and risk destabilizing scratch training on only 400 images), **BOLT-Loss operates entirely at training time through loss formulation**. At test time, the model runs identically to the baseline.

---

## 2. Mathematical Formulation of BOLT-Loss

The overall training objective is a unified tripartite loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{OHEM-wCE}} + \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}$$

where:
- $\mathcal{L}_{\text{OHEM-wCE}}$: Online Hard Example Mining with Class-Frequency Weighting.
- $\mathcal{L}_{\text{Dice}}$: Multi-class Soft-Dice loss on valid classes (from baseline).
- $\mathcal{L}_{\text{bound}}$: Class-Weighted Boundary Loss targeting thin/small objects.
- Hyperparameters: $\lambda_{\text{dice}} = 1.0$, $\lambda_{\text{bound}} = 1.0$.

```
                    ┌────────────────────────────────────────────────────────┐
                    │            Network Output Logits: (B, 19, H, W)        │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
               ┌───────────────────────────────┼──────────────────────────────┐
               ▼                               ▼                              ▼
    ┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
    │  Component B + OHEM │        │   Baseline Dice     │        │     Component A     │
    │  Class-Weighted CE  │        │   Multi-Class Soft  │        │  Class-Weighted     │
    │  + Top-K Hard Pixels│        │   Dice Loss         │        │  Boundary Loss      │
    │  L_OHEM-wCE         │        │   L_Dice            │        │  L_bound            │
    └──────────┬──────────┘        └──────────┬──────────┘        └──────────┬──────────┘
               │                              │                              │
               └──────────────────────► ──(+)─ ◄─────────────────────────────┘
                                       │
                                       ▼
                             Total Loss: L_total
```

---

### Component B: Smoothed Inverse-Frequency Class Weighting

To prevent dominant background classes from overpowering gradients, we compute smoothed inverse-frequency weights derived directly from our Phase 2 training set pixel audit ($N_{\text{total}} = 829,440,000$ pixels):

$$p_c = \frac{N_c}{\sum_{k=0}^{C-1} N_k}$$

To prevent extreme gradient spikes for ultra-rare classes (such as `truck` at 0.02% or `rider` at 0.04%), we apply log-smoothed median frequency balancing:

$$w_c = \frac{1}{\ln(c_{\text{smooth}} + p_c)}$$

where $c_{\text{smooth}} = 1.02$. The weights are normalized such that the mean weight across all 19 classes is 1.0:

$$\tilde{w}_c = C \cdot \frac{w_c}{\sum_{k=0}^{C-1} w_k}$$

#### Resulting Class Weights ($\tilde{w}_c$):
- **Dominant Background (Down-weighted):** `road` = 0.35, `building` = 0.52, `sky` = 0.58, `vegetation` = 0.64.
- **Moderate Infrastructure:** `sidewalk` = 0.76, `wall` = 1.15, `fence` = 1.23, `pole` = 1.28.
- **Rare Foreground & Small Objects (Up-weighted):** `traffic light` = 1.48, `traffic sign` = 1.43, `person` = 1.51, `rider` = 1.55, `car` = 1.10, `truck` = 1.56, `train` = 1.34, `motorcycle` = 1.55, `bicycle` = 1.53.

---

### Component OHEM: Online Hard Example Mining

Standard Cross-Entropy averages loss across all valid pixels in a crop, allowing millions of easily-classified asphalt and sky pixels to dilute the loss on hard vehicle and pedestrian pixels.

For each pixel $i \in \Omega_{\text{valid}}$ (where ground truth $y_i \neq 255$):

$$\ell_i = - \tilde{w}_{y_i} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{c=0}^{C-1} \exp(z_{i, c})} \right)$$

1. Compute per-pixel weighted loss $\ell_i$ for all valid pixels in the batch of size $N_{\text{valid}}$.
2. Sort losses in descending order: $\ell_{(1)} \ge \ell_{(2)} \ge \dots \ge \ell_{(N_{\text{valid}})}$.
3. Retain only the top $K$ hardest pixels:
   $$K = \max\left( \lceil \gamma \cdot N_{\text{valid}} \rceil, K_{\text{min}} \right)$$
   where mining ratio $\gamma = 0.50$ (top 50% hardest pixels retained) and $K_{\text{min}} = 10,000$.
4. Backpropagate only through the hard subset:
   $$\mathcal{L}_{\text{OHEM-wCE}} = \frac{1}{K} \sum_{k=1}^K \ell_{(k)}$$

**Impact:** Easy pixels contributing trivial gradient are masked out; 100% of the backpropagation signal focuses on ambiguous boundaries, underexposed objects, and misclassified foreground pixels.

---

### Component A: Class-Weighted Semantic Boundary Loss ($\mathcal{L}_{\text{bound}}$)

Proposed by the project mentor and motivated by our Phase 8 trimap findings (30.6% accuracy drop on boundaries).

#### Step 1: Spatial Boundary Extraction (Differentiable)
Given ground truth mask $Y \in \{0, \dots, C-1, 255\}^{H \times W}$, a binary boundary map $B \in \{0, 1\}^{H \times W}$ is extracted on-the-fly using 2D max-pooling morphological gradient:

$$B(Y) = \left( \text{MaxPool2D}(Y \cdot \mathbb{I}_{Y \neq 255}, k=3, s=1, p=1) \neq \text{MinPool2D}(Y \cdot \mathbb{I}_{Y \neq 255}, k=3, s=1, p=1) \right) \land (Y \neq 255)$$

This identifies all pixels situated within a 1-pixel radius of any semantic class transition.

#### Step 2: Class-Specific Boundary Multipliers ($\beta_c$)
To specifically boost smaller and underperforming objects (cars, signs, poles, pedestrians), boundary pixels bordering class $c$ receive an enhanced class penalty $\beta_c$:

$$\beta_c = \begin{cases} 
5.0 & \text{if } c \in \{\text{car}, \text{truck}, \text{bus}, \text{person}, \text{rider}, \text{motorcycle}, \text{bicycle}\} \\
3.0 & \text{if } c \in \{\text{traffic light}, \text{traffic sign}, \text{pole}, \text{fence}\} \\
1.0 & \text{otherwise (road, building, sky, vegetation)}
\end{cases}$$

#### Step 3: Differentiable Boundary Loss Formulation
Let $P_c(i) = \text{Softmax}(Z(i))_c$ be the predicted probability of the true class $y_i$ at boundary pixel $i$. The boundary loss penalizes classification error specifically in the boundary band:

$$\mathcal{L}_{\text{bound}} = \frac{\sum_{i \in \Omega_{\text{valid}}} B_i \cdot \beta_{y_i} \cdot (1 - P_{y_i}(i))^2 \cdot (-\log P_{y_i}(i))}{\sum_{i \in \Omega_{\text{valid}}} B_i \cdot \beta_{y_i} + \epsilon}$$

This acts as a focal boundary penalty: boundary pixels of small foreground objects receive up to a $5\times$ gradient boost when misclassified, compelling the SSM encoder-decoder to preserve sharp spatial contours.

---

## 3. Comparison of Approaches & Trade-Off Matrix

| Dimension | Baseline | Mentor's Boundary Loss (A alone) | Class Weighting (B alone) | Proposed BOLT-Loss (A + B + OHEM) |
|---|---|---|---|---|
| **Inference Params** | 26,975 (0.027M) | 26,975 (0.027M) | 26,975 (0.027M) | **26,975 (0.027M)** |
| **Inference FLOPs** | 1.02 G (1080p) | 1.02 G (1080p) | 1.02 G (1080p) | **1.02 G (1080p)** |
| **Targets Boundary Drop?** | ❌ No | ✅ Yes (directly) | ❌ No | ✅ **Yes (via $\mathcal{L}_{\text{bound}}$)** |
| **Targets 0% Foreground?** | ❌ No | ⚠️ Partial (cannot refine what is never detected) | ⚠️ Partial (reweights, but easy pixels still dominate) | ✅ **Yes (OHEM forces gradient onto rare classes)** |
| **Hyperparameter Safety** | Stable | Stable | Sensitive if weights too large | **Stable (log-smoothed weights + 50% OHEM clamp)** |
| **Implementation Scope** | `src/losses.py` | `src/losses.py` | `src/losses.py` | `src/losses.py` + config flag |

---

## 4. Falsifiable Hypotheses & Success Criteria

1. **Overall mIoU Hypothesis:**  
   BOLT-Loss will improve overall test mIoU from **20.70%** (baseline) to **$\ge 24.0\%$** under identical 300-epoch, $lr=0.01$ training.
2. **Boundary Precision Hypothesis:**  
   3-pixel boundary band pixel accuracy will improve from **44.65%** to **$\ge 50.0\%$**.
3. **Dynamic Foreground Object Emergence Hypothesis:**  
   IoU on the `car` and `person` categories will rise from **0.0%** to **$\ge 8.0\%$**, breaking the catastrophic zero-recall failure mode.
4. **Efficiency Invariance:**  
   Parameters (26,975), 1080p FLOPs (1.02 GFLOPs), and inference FPS (~87 FPS on T4 GPU) will remain **strictly identical** to the baseline.

---

## 5. Experiment & Ablation Plan (Phase 11 & 12)

To rigorously isolate the contribution of each component, we define 4 modular experiment configurations:

| Config ID | Description | CE Weighting ($\tilde{w}_c$) | OHEM ($\gamma=0.5$) | Boundary Loss ($\lambda_{\text{bound}}$) |
|---|---|---|---|---|
| `exp_baseline` | Baseline reproduction | Uniform (1.0) | Disabled (1.0) | 0.0 |
| `exp_ablation_wce` | Class Weighting alone | Smoothed $\tilde{w}_c$ | Disabled (1.0) | 0.0 |
| `exp_ablation_ohem` | Class Weighting + OHEM | Smoothed $\tilde{w}_c$ | Enabled (0.5) | 0.0 |
| `exp_bolt_full` | Full Proposed BOLT-Loss | Smoothed $\tilde{w}_c$ | Enabled (0.5) | 1.0 (Class-Weighted $\beta_c$) |
