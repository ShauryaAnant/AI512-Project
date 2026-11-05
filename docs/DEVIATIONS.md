# Deviations Log

This document records every modification or deviation from the supervisor's initial specification or original reference codebase, per Rule 10 of `ROADMAP.md`.

---

### DEV-001: Output Head Returns Raw Logits (Removed Sigmoid)
- **Date:** 2026-11-05
- **Files Affected:** `src/models/model.py`
- **What Changed:** The original `model.py` returned `torch.sigmoid(out0)` for binary segmentation. In `src/models/model.py`, the model returns raw unnormalized logits `out0` of shape `(B, 19, H, W)`.
- **Reason:** For 19-class semantic segmentation, standard PyTorch loss functions (`nn.CrossEntropyLoss`) expect unnormalized class logits. Applying sigmoid to multi-class logits produces incorrect gradient updates and prevents proper softmax/cross-entropy computation.
- **Evidence:** Required by standard multi-class segmentation protocol in `ROADMAP.md` Phase 1.

---

### DEV-002: Native PyTorch Fallback for `trunc_normal_`
- **Date:** 2026-11-05
- **Files Affected:** `src/models/model.py`
- **What Changed:** Added `try / except` to fall back from `timm.models.layers.trunc_normal_` to `torch.nn.init.trunc_normal_`.
- **Reason:** `torch.nn.init.trunc_normal_` is standard in modern PyTorch (≥1.7) and eliminates an unnecessary hard runtime crash if `timm` is not yet installed in a lightweight environment.
- **Evidence:** Verified mathematically equivalent tensor initialization in `tests/test_model.py`.
