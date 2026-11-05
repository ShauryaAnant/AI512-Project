# Hardware and Environment Log

## Local Machine (Development & CPU Verification)
- **OS:** Windows 11 / Windows x86_64
- **Python Version:** Python 3.12.10 (located at `C:\Users\shaur\AppData\Local\Python\pythoncore-3.12-64\python.exe`, configured in local `.venv`)
- **PyTorch Version:** 2.14.1+cpu
- **Role:** Unit tests, dataset audit scripts, metrics validation, documentation, git tracking.

## Kaggle Environment (GPU Training & Baseline Reproduction)
- **Target GPU:** NVIDIA T4 (16GB VRAM) or P100 (16GB VRAM)
- **Execution Strategy:** Private Kaggle notebook with git-cloned repository and mounted private ACDC Night dataset.
- **Precision:** FP32 or FP16 AMP (after numerical stability check on SSM).
- **Driver / CUDA Details:** Will be recorded upon first Kaggle GPU execution in `docs/EXPERIMENT_LOG.md`.
