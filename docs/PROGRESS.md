# Project Progress Checklist

| Phase | Description | Status | Date Completed | Key Artifacts / Outputs |
|---|---|---|---|---|
| **Phase 0** | Environment & repo bootstrap | ✅ | 2026-11-05 | `.gitignore`, directory structure, `.venv` with Python 3.12.10, PyTorch 2.14.1, `docs/vmunet_notes.md`, `docs/hardware.md` |
| **Phase 1** | Understand & integrate supplied model | ✅ | 2026-11-05 | `src/models/model.py` (`build_model`), `tests/test_model.py`, `docs/model_notes.md`, `docs/DEVIATIONS.md` |
| **Phase 2** | ACDC night data audit | ✅ | 2026-11-05 | `scripts/audit_dataset.py`, `docs/dataset_audit.md`, `data/processed/illum_stats.csv`, `docs/figures/` |
| **Phase 3** | Evaluation protocol & split generation | ✅ | 2026-11-07 | `scripts/make_splits.py`, `splits/*.txt`, `docs/protocol.md` |
| **Phase 4** | Dataset loader & preprocessing | ✅ | 2026-11-07 | `src/datasets/acdc.py`, `src/datasets/transforms.py`, `tests/test_dataset.py` |
| **Phase 5** | Loss, metrics & efficiency tooling | ⏳ Pending | | `src/losses.py`, `src/metrics.py`, `src/utils/flops.py`, `scripts/benchmark_speed.py` |
| **Phase 6** | `train.py` and `inference.py` | ⏳ Pending | | `train.py`, `inference.py`, smoke test |
| **Phase 7** | Baseline reproduction (ACDC Night, 3 seeds) | ⏳ Pending | | `results/baseline/*.json`, `docs/baseline_results.md` |
| **Phase 8** | Failure-case & efficiency analysis | ⏳ Pending | | `scripts/analyze_failures.py`, `docs/failure_analysis.md` |
| **Phase 9** | Literature review & research gap | ⏳ Pending | | `docs/literature_review.md`, `docs/research_gap.md` |
| **Phase 10** | Novel method design | ⏳ Pending | | `docs/method.md` |
| **Phase 11** | Implementation of proposed method | ⏳ Pending | | `src/models/novel_*.py`, unit tests |
| **Phase 12** | Experiments & ablations | ⏳ Pending | | `docs/ablations.md`, multi-seed results |
| **Phase 13** | Report & submission package | ⏳ Pending | | `report/`, final codebase |
