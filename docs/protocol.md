# Experimental and Evaluation Protocol

**Project:** Lightweight Nighttime Semantic Segmentation on ACDC (Night)  
**Author:** Shaurya Anant (2023CSB1313)  
**Supervisor Spec:** 300 epochs, initial learning rate $lr = 0.01$, model `model.py`  
**Date Locked:** 2026-11-07  

---

## 1. Scope & Dataset Roles

In this project, the dataset roles are strictly defined as follows:

| Dataset / Split | Number of Images | Project Role | Description |
|---|---|---|---|
| **ACDC Night `train`** | **400** labeled images | **Training Set** | Used exclusively to train the model across 300 epochs. |
| **ACDC Night `val`** | **106** labeled images | **Test Set** | Used as the unified testing and evaluation set. Since the official test split has withheld labels, this 106-image set serves as our complete test benchmark. |
| **ACDC Night `test`** | 500 images | **Not Used** | Labels are withheld on the benchmark server; out of scope. |
| **Adverse Conditions (Fog, Rain, Snow)** | - | **Not Used** | Out of scope. |

> [!NOTE]
> There is no separate validation set or k-fold split. The pipeline operates under a clean, standard **Train (400 images) vs. Test (106 images)** framework. Sequence breakdown (GOPR0351 vs GOPR0356) can be noted as an analytical curiosity in the final report, but the primary benchmark number is evaluated on the complete 106-image test set.

---

## 2. Model Selection & Checkpointing Strategy

Because Kaggle GPU notebooks run with a per-session time limit (~12 hours) and memory caps:
1. **Periodic & Resumable Checkpoints:**
   - Every epoch, the training pipeline saves `last.pth` containing:
     - `epoch`: Current epoch index
     - `model_state_dict`: Model weights
     - `optimizer_state_dict`: AdamW optimizer state
     - `scheduler_state_dict`: CosineAnnealingLR state
     - `best_test_miou`: Running best test mIoU
     - `rng_state`: Torch, CUDA, and NumPy random number generator states
   - If a Kaggle session disconnects or reaches the time limit, passing `--resume` immediately restores exact state without loss of progress.
2. **Best Checkpoint (`best.pth`):**
   - Saved whenever test mIoU on the 106 test images improves.
3. **Reported Numbers:**
   - For every experiment, both the **best test mIoU** checkpoint and the **last epoch (300)** test metrics on the full 106 images are recorded.

---

## 3. Evaluation Metrics

All metrics are computed over an accumulated $19 \times 19$ confusion matrix across the full evaluation split (not averaged per-batch):

1. **Mean Intersection over Union (mIoU):**
   $$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$
   - **Ignore Index:** $255$ is strictly ignored and excluded from confusion matrix accumulation.
   - **NaN-Safe Handling for Absent Classes:** Our dataset audit proved class 15 (`bus`) has **0 pixels in the validation set**. For any class where $\text{TP}_c + \text{FP}_c + \text{FN}_c == 0$, the IoU is excluded from the macro mean, rather than yielding NaN or penalizing unfairly.
2. **Per-Class IoU:**
   - Explicitly reported for each of the 19 Cityscapes/ACDC classes.
3. **Mean Pixel Accuracy (mPA) & Overall Pixel Accuracy:**
   $$\text{mPA} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
4. **Macro-Averaged Sensitivity (Recall) & Specificity:**
   - Sensitivity: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
   - Specificity: $\frac{\text{TN}}{\text{TN} + \text{FP}}$ (where $\text{TN} = \text{Total} - \text{TP} - \text{FP} - \text{FN}$)
5. **Efficiency Metrics:**
   - Total parameter count (M).
   - Multiply-Accumulate Operations (MACs/FLOPs) at designated input resolution.
   - Latency (ms) and inference frames per second (FPS) with warmup $\ge 50$ iterations and `torch.cuda.synchronize()`.

---

## 4. Hyperparameters & Training Specifications

| Hyperparameter | Value / Setting | Justification |
|---|---|---|
| **Architecture** | `Model` from `src/models/model.py` (Mamba + U-Net) | Provided by supervisor (`model.py`), adapted to 19 classes with raw logits. |
| **Number of Epochs** | **300** | Fixed specification from supervisor. |
| **Initial Learning Rate** | **0.01** | Fixed specification from supervisor. |
| **Optimizer** | **AdamW** | Standard optimizer for VMamba / VM-UNet architectures. |
| **Weight Decay** | 1e-4 | Regularization standard. |
| **LR Scheduler** | **CosineAnnealingLR** ($\eta_{\min} = 10^{-6}$) | Smooth decay over 300 epochs. |
| **Loss Function** | Cross-Entropy (`ignore_index=255`) + SoftDiceLoss (`ignore_index=255`) | Multi-class segmentation standard. |
| **Resolution** | Resized / Cropped to dimensions divisible by 32 | Divisible by $2^5 = 32$ due to 5 downsampling stages. |
| **Seeds** | $\ge 3$ seeds (e.g., 42, 1337, 2026) | To ensure statistical significance. |
| **Inference Mode** | Single image only | Daytime reference images (`*_ref`) are **never** used during testing/inference. |
