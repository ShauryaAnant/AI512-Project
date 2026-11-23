# Project Progress Checklist

| Phase | Description | Status | Date Completed | Key Artifacts / Outputs |
|---|---|---|---|---|
| **Phase 0** | Environment & repo bootstrap | ✅ | 2026-11-05 | `.gitignore`, directory structure, `.venv` with Python 3.12.10, PyTorch 2.14.1, `docs/vmunet_notes.md`, `docs/hardware.md` |
| **Phase 1** | Understand & integrate supplied model | ✅ | 2026-11-05 | `src/models/model.py` (`build_model`), `tests/test_model.py`, `docs/model_notes.md`, `docs/DEVIATIONS.md` |
| **Phase 2** | ACDC night data audit | ✅ | 2026-11-05 | `scripts/audit_dataset.py`, `docs/dataset_audit.md`, `data/processed/illum_stats.csv`, `docs/figures/` |
| **Phase 3** | Evaluation protocol & split generation | ✅ | 2026-11-07 | `scripts/make_splits.py`, `splits/*.txt`, `docs/protocol.md` |
| **Phase 4** | Dataset loader & preprocessing | ✅ | 2026-11-07 | `src/datasets/acdc.py`, `src/datasets/transforms.py`, `tests/test_dataset.py` |
| **Phase 5** | Loss, metrics & efficiency tooling | ✅ | 2026-11-07 | `src/losses.py`, `src/metrics.py`, `src/utils/flops.py`, `scripts/benchmark_speed.py`, `tests/test_metrics.py` |
| **Phase 6** | `train.py` and `inference.py` | ✅ | 2026-11-08 | `train.py`, `inference.py`, `src/engine.py`, `configs/baseline_acdc_night.yaml`, `tests/test_smoke_train.py` |
| **Phase 7** | Baseline reproduction (ACDC Night) | ✅ | 2026-11-09 | Seed 42 run complete, `docs/baseline_results.md` |
| **Phase 8** | Failure-case & efficiency analysis | ✅ | 2026-11-11 | `scripts/analyze_failures.py`, `docs/failure_analysis.md`, `docs/figures/confusion_matrix.png`, `docs/figures/worst_failures_gallery.png` |
| **Phase 9** | Literature review & research gap | ✅ | 2026-11-11 | `docs/literature_review.md`, `docs/research_gap.md` |
| **Phase 10** | Novel method design | ✅ | 2026-11-22 | `docs/method.md` (BOLT-Loss & Frequency-Aware RCS design, zero inference cost) |
| **Phase 11** | Implementation of proposed method | ✅ | 2026-11-22 | `src/losses.py`, `src/datasets/acdc.py` (RCS), `configs/novel_*.yaml`, unit tests verified |
| **Phase 12** | Experiments & ablations | ✅ | 2026-11-23 | Baseline vs. BOLT vs. Boundary ($\lambda=0.2$) vs. RCS documented in `docs/novelty_results.md` |
| **Phase 13** | Report & submission package | ⏳ In Progress | | Preparing final report and comparative figures |
