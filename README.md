# Lightweight Nighttime Semantic Segmentation on ACDC (Night)

This repository implements lightweight nighttime semantic segmentation on the ACDC (Adverse Conditions Dataset with Correspondences) Night benchmark using a hybrid Mamba / State-Space U-Net architecture.

## Repository Structure
```
nightseg-acdc/
├── README.md                   # Setup and execution guide
├── ROADMAP.md                  # Project specification, milestones and rules
├── configs/                    # Experiment configuration files
├── data/                       # Data caches (gitignored)
├── dataset/                    # ACDC dataset root (gitignored)
├── docs/                       # Progress, audit, notes, deviations, and logs
│   ├── PROGRESS.md
│   ├── DEVIATIONS.md
│   ├── hardware.md
│   ├── vmunet_notes.md
│   └── model_notes.md
├── report/                     # Final project report
├── scripts/                    # Audit, split generation, and analysis scripts
├── splits/                     # Split definition files
├── src/
│   ├── models/                 # Model implementations (baseline and novel variants)
│   ├── datasets/               # ACDC dataset loaders and transforms
│   ├── utils/                  # Metrics, FLOPs, logging, and seed helpers
│   ├── losses.py               # Multi-class Cross-Entropy and SoftDice loss
│   └── engine.py               # Training and validation loops
├── tests/                      # Unit tests
├── train.py                    # Training entry point
└── inference.py                # Inference and prediction exporter
```

## Quick Start

### 1. Environment Setup
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Run model unit tests
$env:PYTHONPATH="."
python tests/test_model.py
```
