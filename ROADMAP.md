# Lightweight Nighttime Semantic Segmentation on ACDC (Night) — Project Roadmap

**Owner:** Shaurya Anant (2023CSB1313) · **Partner (separate dataset, NightCity):** Kanav Puri (2023MCB1298)
**Supervisor:** Sridhatta Jayaram Aithal
**Audience:** the AI-enabled IDE / coding assistant working in this repo. Read this whole file before writing any code. Re-read §1 (Rules) before every task.
**Revision:** v3, 5 Nov 2026 — dataset changed from Dark Zurich to **ACDC** (see Appendix C for what changed and why).

---

## 0. Project in one paragraph

Take the **supplied model** (`model.py`, VM-UNet-style Mamba/VSS U-Net, provided via Google Drive — the owner must place it in the repo; do not invent or re-implement it), build a training / inference / evaluation pipeline for **ACDC night** by adapting the official VM-UNet repo (https://github.com/JCruan519/VM-UNet, originally *binary medical* segmentation → we need *19-class Cityscapes-style* segmentation), **reproduce a baseline**, analyse failures, review literature, identify a research gap, **propose and validate a novel method** with comparisons and ablations, and write a report. Initial config from the supervisor: **300 epochs, lr = 0.01, model file `model.py`**.

**Scope:** only the **night** condition of ACDC. Fog / rain / snow folders exist in the download but are **not used** unless the owner explicitly approves an extra experiment (log it in `docs/DEVIATIONS.md`).

### ACDC facts VERIFIED from the owner's directory tree (5 Nov 2026; `scripts/audit_dataset.py` must re-confirm every number)

**Layout (note the double nesting, `gt/gt/` and `rgb_anon/rgb_anon/`):**
```
<root>/gt/            License.pdf  README.md  gt/<cond>/<split>/<SEQ>/<SEQ>_frame_<N>_gt_{labelTrainIds,labelIds,labelColor,invIds,invGray}.png
<root>/rgb_anon/      License.pdf  README.md  rgb_anon/<cond>/{train,val,test,train_ref,val_ref,test_ref}/<SEQ>/
                      images:  <SEQ>_frame_<N>_rgb_anon.png        reference (day) images: <SEQ>_frame_<N>_rgb_ref_anon.png
<cond> ∈ {fog, night, rain, snow}      →  we use  night  only
```

**Night subset (counts parsed from the tree):**

| Split | Images | Sequences (images) | Labels |
|---|---|---|---|
| `train` | **400** | GOPR0351 (93), GOPR0376 (147), GP010376 (56), GP010397 (60), GP020397 (44) | **Yes** (`gt/.../night/train`) |
| `val` | **106** | GOPR0351 (56), GOPR0356 (50) | **Yes** (`gt/.../night/val`) |
| `test` | **500** | GOPR0355 (10), GOPR0356 (179), GOPR0364 (105), GOPR0594 (148), GP010364 (46), GP010594 (12) | **No — withheld** (no `gt/.../night/test` folder) |
| `train_ref` / `val_ref` / `test_ref` | 400 / 106 / 500 | same sequences, 1-to-1 with the images above | **No labels in this download** |

These numbers (400 / 106 / 500) agree with the ACDC paper (Sakaridis et al., ICCV 2021, arXiv 2104.13395: night = 400 train / 106 val / 500 test).

**Consequences (these override anything below that conflicts):**
1. **Supervised training is valid again.** 400 labeled night training images + 106 labeled night validation images. The supervisor's "300 epochs, lr 0.01" applies directly to the 400-image train split.
2. **Official splits are used as-is** (train / val; no k-fold, no re-splitting). Never move images between splits.
3. **Test labels are withheld.** The only way to get a test number is the ACDC benchmark server (https://acdc.vision.ee.ethz.ch/, account needed). Submission format, quotas and returned metrics are **not verified** → the owner reads the site/README and records them in `docs/protocol.md` before any submission (see Phase 3/7). Until then, the headline number is on **val**.
4. **Same-drive leakage risk inside the official split (measured from frame indices):** sequence **GOPR0351 appears in both train (frames 159–639) and val (frames 33–825)**; the closest val↔train pair of that sequence is only **12 frames apart**. Sequence **GOPR0356 is in val and in test but NOT in train**; test sequences GOPR0355, GOPR0364, GOPR0594, GP010364, GP010594 are not in train either. Therefore: report val **overall (106)** AND two sub-sets: **val-seen-drive = GOPR0351 (56)** and **val-unseen-drive = GOPR0356 (50)**. The unseen-drive subset is the more honest generalization estimate. Do not hide the gap if the two differ.
5. **Invalid-region masks exist** (`*_gt_invIds.png`, `*_gt_invGray.png`; ACDC defines a binary mask of regions with uncertain semantics). Main metric stays standard mIoU on `labelTrainIds` (255 = ignore). Use the invalid masks for **analysis** (Phase 8: valid vs invalid region performance). Confirm their exact encoding from `gt/README.md` — do not guess.
6. **Reference (day) images** (`*_ref`) show the same place in normal conditions (GPS-matched, **not pixel-aligned**; dynamic objects differ). They are unlabeled here. They may be used **only at training time** (never at inference, which must stay single-image) and only after the owner approves it as a method component (Phase 10, candidate 4).
7. **There is no large unlabeled night pool** (unlike Dark Zurich). Do not plan self-training on thousands of unlabeled night frames.
8. **Licensing:** read `License.pdf` (in both `gt/` and `rgb_anon/`). Keep any Kaggle copy of ACDC **private**; do not redistribute images, labels or predictions-with-images publicly; cite the ACDC paper.

### Dark Zurich is dropped (supervisor's decision) — and must not be used as an "independent" test
From the two directory trees, the Dark Zurich night frames are **the same drives and the same frame indices** as ACDC night: DZ val night `GOPR0356` (50 frames, indices 321–488) = ACDC val `GOPR0356` (identical index list); DZ test night `GOPR0364` (105) + `GP010364` (46) = the 151 ACDC test frames of those two sequences (identical index lists); DZ train night sequences `GOPR0351/GOPR0376/GP010376` contain every ACDC labeled frame index of those sequences. (Matching is by sequence name + frame index from the listings; confirm by image hash only if the owner ever wants to use DZ.) Consequence: **evaluating an ACDC-trained model on Dark Zurich is NOT a held-out test.** Dark Zurich can be dropped completely from the repo; keep the earlier download only as a reference.

### Literature reference numbers for sanity checks (ACDC paper, supervised, night, mIoU %, trained on ACDC night train)
RefineNet 52.2 · DeepLabv2 45.4 · DeepLabv3+ 59.2 · HRNet 63.2 (also: models trained on all adverse conditions score 55.5 / 45.3 / 60.9 / 65.3 on night). These are **different architectures/resolutions/protocols** (and are test-server numbers). Use them only as an order-of-magnitude sanity check (a lightweight model landing far below ~40 or far above ~70 on val deserves a bug hunt first). Fetch current leaderboard numbers from the website before quoting any "state of the art" in the report; do not copy numbers from memory.

---

## 0.5 COMPUTE: KAGGLE

All training/eval runs on **Kaggle notebooks (GPU)**. The IDE cannot run Kaggle directly: edit code in the repo, push to GitHub, and the owner (or `kaggle kernels push`) runs a thin notebook that clones the repo and calls `train.py` / `inference.py`. Rules:
1. **Data on Kaggle:** ACDC is not on Kaggle's public catalog under the owner's control → the owner uploads a **private Kaggle Dataset** containing **only the night subset** (`gt/gt/night/{train,val}`, `rgb_anon/rgb_anon/night/{train,val,test,train_ref,val_ref,test_ref}`, plus both `README.md` and `License.pdf`), preserving the double-nested layout. It is mounted read-only at `/kaggle/input/<dataset-slug>/` (confirm the exact path in the audit). Outputs go to `/kaggle/working/` (~20 GB cap). All roots live in YAML (`data_root`, `out_dir`); never hard-code.
2. **Session limits:** ≈12 h per GPU session and a weekly GPU-hour quota (check current numbers in the Kaggle UI). Everything must be **resumable**: save `last.pth` (model+optimizer+scheduler+epoch+RNG) every N epochs, auto-resume from `out_dir`, and persist outputs ("Save Version" or upload as a Kaggle Dataset) before a session ends. Log elapsed time per epoch to plan 300-epoch runs.
3. **GPU choice:** prefer **T4**, and verify `mamba_ssm` / selective-scan CUDA kernels build and run (P100 is older, sm_60 — may not work; test first and record the outcome). T4 has no bf16: use fp32, or fp16 AMP only after checking the SSM does not produce NaNs. Record GPU name and driver in `docs/hardware.md`.
4. **Setup cell:** pin install commands for `mamba_ssm`, `causal-conv1d` and PyTorch in `kaggle/setup.sh`; internet must be enabled in the notebook to pip/git clone. Document any wheel/compile workaround that actually worked.
5. **Data I/O:** ACDC frames are high-resolution PNGs (**verify exact size in the audit**; they are GoPro 1080p captures, so expect 1920×1080). Decoding PNGs every epoch is slow: pre-resize **once** (train/val images with bilinear, labels with nearest) to the training resolution and cache to `/kaggle/working/cache/`; `num_workers` ≤ 4. 400 train images × 300 epochs is cheap per run; the expensive part is repeats (seeds × ablations). Benchmark one epoch before launching sweeps.
6. Local PC (CPU) is for scripts that need no GPU: dataset audit, unit tests, plotting.

---

## 1. RULES (non-negotiable, apply to every task)

### 1.1 Honesty & scientific integrity
1. **Never fabricate** numbers, results, logs, dataset statistics, citations or paper claims. If something wasn't run, say "not run". Placeholders in tables must be literally `TBD`.
2. **Verify, don't assume**: dataset structure, label IDs, class counts, image sizes, annotation availability must be measured by script and saved to `docs/dataset_audit.md`.
3. Every citation must be a real paper the owner can open; list URL/arXiv ID. If unsure a paper exists, flag it as "unverified" — never guess.
4. Report **negative results and failures** too.
5. If the specified config misbehaves (e.g. lr 0.01 diverges), **document the observation first, then ask the owner** before changing it. Do not silently change hyperparameters.
6. **Selection hygiene:** val is used for checkpoint selection, so val numbers are optimistic. Always also report the **last-epoch** val result and the **val-unseen-drive** subset (Consequence 4). Test images: predictions only, never used for any decision.

### 1.2 Change control
7. **Minimal, targeted changes** over rewrites. Do not refactor unrelated files. Prefer small diffs; explain each.
8. Never modify raw data in `data/raw/`. Derived data goes to `data/processed/`.
9. Never modify the supplied `model.py` architecture silently. Baseline copy is frozen at `src/models/model_baseline.py` (read-only after first commit). Novel variants go in separate files.
10. Every deviation from the supervisor's spec → append an entry to `docs/DEVIATIONS.md` (what, why, date, evidence).
11. Ask the owner before: downloading large datasets, installing CUDA-specific packages, deleting files, starting runs longer than 30 min, changing the protocol, **submitting to the ACDC test server**.

### 1.3 Code quality
12. Python ≥3.10, **type hints + docstrings (Google style)** on every public function/class; short module docstring stating purpose and I/O shapes.
13. **No hard-coded paths or hyperparameters.** All config via YAML in `configs/` (+ CLI overrides). One config file fully reproduces one run.
14. **Reproducibility:** set seeds (python, numpy, torch, cudnn flags) from config; log seed, git commit hash, config, package versions, GPU name, CUDA version at start of each run.
15. Use `logging` (not bare `print`), write logs to `runs/<exp_name>/train.log` + TensorBoard (or CSV) scalars.
16. Tensor shapes commented at key points (`# (B, C, H, W)`).
17. Lint/format: `ruff` + `black` (line length 100). Run before every commit.
18. Add **unit tests** (`pytest`) for: dataset loader (shapes, label range, ignore index, image↔label pairing), metrics (against hand-computed confusion matrices), loss (finite, ignores 255), model forward (output `(B,19,H,W)`).
19. Fail loudly: assert shapes, label range ∈ {0..18, 255}, no NaN in loss; abort run on NaN and log it.

### 1.4 Documentation (mandatory, kept up to date as you go)
20. `README.md`: setup, how to train, how to infer, how to reproduce each table.
21. `docs/EXPERIMENT_LOG.md`: one entry **per run**: date, exp name, config path, git hash, goal, hardware, result summary, observations, next step.
22. `docs/DEVIATIONS.md`, `docs/dataset_audit.md`, `docs/protocol.md`, `docs/failure_analysis.md`, `docs/literature_review.md`, `docs/research_gap.md`, `docs/method.md`, `docs/ablations.md`.
23. After finishing every phase: update `docs/PROGRESS.md` (phase checklist with ✅/⏳, date, link to artifacts) and make a git commit.
24. **Git:** feature branch per phase (`phase-3-protocol`), conventional commits (`feat:`, `fix:`, `docs:`, `exp:`), never commit data, checkpoints, or large files (use `.gitignore`; track results JSON/CSV and small figures only). **Never commit ACDC images or labels** (license).

### 1.5 Experiment discipline
25. One change per experiment (ablations must be clean). Name runs `{phase}_{model}_{change}_{seed}`.
26. Each run saves: `config.yaml`, `train.log`, `metrics.csv`, `best.pth`, `last.pth`, `results.json`, predicted maps for val (and test if inference was run) in `preds/`.
27. Every result reported in the report must be traceable to a `results.json` + run ID.
28. Report mean ± std over **≥3 seeds** for baseline and final method (feasible now: 400 images). State clearly when only 1 seed.
29. Test split: inference only, produce predictions once per final model; no tuning based on server scores.

### 1.6 Shared reporting with Kanav
30. Use the **common results schema** (§Appendix A) so both datasets' numbers can be reviewed consistently. Same metric code semantics, same table layouts.

---

## 2. Target repo structure

```
nightseg-acdc/
├── README.md
├── ROADMAP.md                  # this file
├── requirements.txt / environment.yml
├── configs/
│   ├── base.yaml
│   ├── baseline_acdc_night.yaml
│   └── novel_*.yaml
├── data/                       # gitignored
│   ├── raw/acdc/               # untouched download (or Kaggle mount)
│   └── processed/              # resized caches
├── splits/                     # official split lists generated from folder listing (committed, small; no images)
├── src/
│   ├── models/{model_baseline.py, model.py, novel_*.py}   # model.py = integration entry point
│   ├── datasets/{acdc.py, transforms.py}
│   ├── losses.py   metrics.py   utils/{seed.py, logger.py, ckpt.py, viz.py, flops.py}
│   └── engine.py               # train_one_epoch / validate (adapted from VM-UNet)
├── scripts/{audit_dataset.py, make_splits.py, benchmark_speed.py, analyze_failures.py, export_test_preds.py}
├── train.py   inference.py     # required deliverables
├── tests/
├── runs/                       # gitignored except results.json
├── docs/
└── report/
```

---

## 3. PHASES

Each phase lists **Goal → Tasks → Deliverables → Done-when**. Do not start a phase until the previous "Done-when" is met (except where noted as parallelizable).

### Phase 0 — Environment & repo bootstrap
**Goal:** reproducible environment; official VM-UNet understood.
- Create repo skeleton (§2), `.gitignore`, pre-commit (ruff/black), `docs/PROGRESS.md`.
- Clone official VM-UNet **as read-only reference** into `third_party/VM-UNet/` (do not edit; copy and adapt pieces into `src/`). Read: `train.py`, `engine.py`, `configs/config_setting.py`, `datasets/dataset.py`, `models/vmunet/`, `utils.py`.
- Write `docs/vmunet_notes.md` summarizing: optimizer (AdamW default), scheduler (CosineAnnealing), loss (BceDiceLoss), input size, batch size, pretrained weights loading (VMamba-S ImageNet checkpoint), metrics, checkpoint logic. **List what is binary-specific and must change** for 19-class segmentation.
- Environment: PyTorch + CUDA matching the Kaggle GPU; `mamba_ssm` and `causal-conv1d` are CUDA-version sensitive → pin versions in `requirements.txt`, document the exact install commands that worked.
- Record hardware (GPU model, VRAM, CPU, RAM) in `docs/hardware.md`.
- **Owner action:** download `model.py` (+ any weights) from the supervisor's Google Drive folder into `src/models/`.
**Done-when:** `python -c "import torch, mamba_ssm"` works on Kaggle; reference repo notes written.

### Phase 1 — Understand & integrate the supplied model
**Goal:** `model.py` imports, runs forward/backward, shape contract known.
- Read `model.py` fully. Document in `docs/model_notes.md`: class names, constructor args, expected input size/channels, output shape, how many classes/final head, pretrained weight expectations, blocks (VSS, SS2D, patch merge/expand, skip connections), any hard-coded 1-channel/binary assumptions.
- Freeze a copy as `model_baseline.py`; expose `build_model(cfg)` in `src/models/model.py` with `num_classes=19`, `input_channels=3`.
- Make the head output **raw logits** `(B,19,H,W)` (remove sigmoid if binary head). Document any such change in DEVIATIONS.md.
- Test: forward + backward on a dummy `(2,3,H,W)` tensor at several resolutions; check divisibility constraints (H, W multiples of 32?); check VRAM at target crop size and batch size.
**Done-when:** unit test `test_model_forward` passes; shape & memory table in docs.

### Phase 2 — ACDC night data audit (CRITICAL)
**Goal:** confirm every fact in §0 by script, on the real files. Output facts, not assumptions.
- `scripts/audit_dataset.py` walks `<data_root>` and reports for **night only**: tree to depth 4; counts per split/sequence (compare against the §0 table; any mismatch is a finding); image size(s) (full scan of unique sizes); label PNG dtype/mode and **unique pixel values** (+ counts) per file type (`labelTrainIds` must be ⊂ {0..18, 255}; `labelIds` is raw Cityscapes ids 0–33 → never train on it); **image↔label pairing** (`*_rgb_anon.png` ↔ `*_gt_labelTrainIds.png`, report unmatched); pairing of each image with its `*_rgb_ref_anon.png`; the encoding of `invIds`/`invGray` (cross-check with `gt/README.md`); class histogram on train and on val (log scale plot); % ignore pixels; % invalid-mask pixels; corrupt/duplicate files; sequence overlap table (train/val/test) reproducing Consequence 4.
- Save a visual sanity grid (10 image | label | invalid-mask overlays) to `docs/figures/`.
- Compute simple illumination statistics per image (mean luma, std luma, % saturated pixels, % very dark pixels) and save to `data/processed/illum_stats.csv` — used in Phase 8.
- Read both `README.md` files and `License.pdf`; summarize the relevant rules in `docs/dataset_audit.md`.
- Write `docs/dataset_audit.md` with all findings.
**Done-when:** audit doc committed; counts and label format confirmed; mismatches (if any) reported to the owner.

### Phase 3 — Protocol (short; needs owner + supervisor confirmation)
**Goal:** lock the evaluation protocol in `docs/protocol.md` **before** any long run.

| Item | Default proposal |
|---|---|
| Training data | ACDC **night train (400)**; official split; no extra data; ImageNet/VMamba-pretrained init only if the supplied `model.py` expects it (document) |
| Model selection | best **val mIoU** epoch on night val (106); also report last epoch |
| Reported numbers | val overall (106), val-seen-drive (GOPR0351, 56), val-unseen-drive (GOPR0356, 50); **test via server** only for the final baseline and final proposed model, if the owner confirms account/format/quota |
| Metric | standard mIoU, 19 classes, 255 ignored (+ the extra metrics of Phase 5). AUIoU / uncertainty-aware tracks are **out of scope** |
| Inference | single image only (reference day images are **not** used at inference) |
| Out of scope | fog/rain/snow, UDA (Cityscapes→ACDC), panoptic/detection, Dark Zurich |

**Questions to put to the supervisor (draft message saved in `docs/protocol.md`):** (1) night-only training is acceptable? (2) is test-server submission expected, and how many submissions are allowed? (3) is it acceptable that val (with GOPR0351 shared with train) is used for checkpoint selection, with the unseen-drive subset reported separately? (4) may training-time-only use of the reference day images count as part of our novel method?
**Done-when:** protocol approved by owner; `scripts/make_splits.py` has produced `splits/night_{train,val,test}.txt` (+ `val_seen.txt`, `val_unseen.txt`) deterministically from the folder listing.

### Phase 4 — Dataset loader & preprocessing
- `src/datasets/acdc.py`: `torch.utils.data.Dataset` over the official night splits. Pairing rule: `.../rgb_anon/night/<split>/<SEQ>/<SEQ>_frame_<N>_rgb_anon.png` ↔ `.../gt/night/<split>/<SEQ>/<SEQ>_frame_<N>_gt_labelTrainIds.png`. Returns `image (3,H,W) float`, `mask (H,W) long` in 0..18 or 255, plus filename/sequence id. Test split → image-only mode (no mask). Optional `with_ref=True` returns the reference day image (training-time-only experiments; off by default; used only after approval).
- Native size is large for VM-UNet: choose **train resize + random crop** (e.g. rescale to ~960×540, crop 512×512 or 512×1024; decide after the VRAM test in Phase 1) and **evaluation resolution** — state both clearly. For fair metrics, **upsample logits to the label resolution before argmax**, or evaluate at one fixed, documented resolution for every model (baseline and proposed identical).
- Augmentations (paired image/mask): random scale (0.5–2.0), crop, horizontal flip, mild color jitter/gamma. **No augmentation on val/test.** Masks use nearest interpolation only.
- Normalize with ImageNet mean/std unless `model.py` dictates otherwise (document). Cache resized data to `data/processed/` (Kaggle: `/kaggle/working/cache`).
- Tests: shapes, dtype, label range, ignore handling, image↔label pairing for every file, flip applied identically to image/mask, determinism with fixed seed.
**Done-when:** DataLoader iterates one full epoch without error; sanity visualization saved.

### Phase 5 — Loss, metrics, efficiency tooling
- **Loss:** replace BceDice with multi-class `CrossEntropy(ignore_index=255)` + soft-Dice (multi-class, ignore-aware); weights configurable (`ce_w`, `dice_w`). Optional class weights / OHEM as *later* experiments, not baseline.
- **Metrics (`metrics.py`)** from an accumulated 19×19 confusion matrix over the whole eval set (not per-batch averaging):
  - mIoU, per-class IoU (NaN-safe for classes absent in the eval set; state how handled — `val-unseen` may miss classes, e.g. train/bus/motorcycle)
  - Dice (per-class + mean), mean pixel accuracy (mPA), overall pixel accuracy
  - **Sensitivity (recall)** and **specificity** per class, macro-averaged (one-vs-rest from the confusion matrix: TN = total − TP − FP − FN)
  - All of the above computed for: val overall, val-seen-drive, val-unseen-drive, and (analysis only) valid-mask vs invalid-mask pixels
  - Unit-test against tiny hand-computed matrices.
- **Efficiency (`utils/flops.py`, `scripts/benchmark_speed.py`):** parameters (M); FLOPs/MACs at a stated input resolution — selective-scan ops are **not counted by default** by fvcore/thop; use VMamba's FLOPs helper or add a custom handle for the SSM scan and **state the method used**. Latency (ms) and FPS: warm-up ≥50 iters, `torch.cuda.synchronize()`, ≥200 timed iters, batch size 1, fp32 and (optionally) AMP, report mean±std, and GPU/driver/CUDA/PyTorch versions.
**Done-when:** all metric tests pass; benchmark script prints a table.

### Phase 6 — `train.py` and `inference.py`
- **train.py** (adapted from VM-UNet engine): config-driven; AdamW (as in VM-UNet) **starting at the specified lr = 0.01**, CosineAnnealing, **300 epochs**, AMP optional, grad-clip configurable, gradient norm + loss logged each epoch, validation every epoch (106 images is cheap; if too slow, every N epochs), best checkpoint by val mIoU **and** `last.pth` every epoch (resume support: model, optimizer, scheduler, epoch, best metric, RNG state).
- **Stability monitoring** (per supervisor): log train/val loss curves, lr, grad-norm; auto-detect NaN/divergence and stop. If lr = 0.01 is unstable, **write an observation entry (plots + numbers) in EXPERIMENT_LOG.md and DEVIATIONS.md, then ask the owner** before trying alternatives (e.g. 1e-3, warm-up, grad-clip) — and keep the failed run.
- **inference.py:** loads checkpoint + config; runs on a split (`val` / `test`) or a folder; saves predicted label maps (single-channel `trainId` PNG at the **original image resolution**, filename = image stem), color-coded overlays, and `results.json` (all Phase 5 metrics + efficiency numbers; test split → predictions only, no metrics). Optional flip/multi-scale TTA flag, off by default.
- `scripts/export_test_preds.py`: package test predictions in the format the ACDC server requires (**owner confirms the format first**, Phase 3). Never submit without the owner's go-ahead.
- Smoke test: overfit 4 images for 100 iterations → loss must drop clearly. Do this before any long run.
**Done-when:** smoke test passes; a 2-epoch run completes end-to-end including resume.

### Phase 7 — Baseline reproduction
- Run `lr=0.01`, 300 epochs on night train, **3 seeds**. Record everything per Rule 26.
- Fill the **baseline results table** (Appendix A schema): mIoU, per-class IoU, Dice, mPA, sensitivity, specificity (val overall / seen-drive / unseen-drive, best and last epoch), params, FLOPs, latency, FPS, hardware.
- Document optimizer, loss, input resolution, preprocessing, hardware, any deviations.
- Save checkpoints, logs, predicted maps (val; and test predictions for the median seed).
- If the owner approves: submit the baseline's test predictions to the ACDC server once and record the returned numbers in `docs/baseline_results.md` (with date and submission ID).
- Compare to the literature reference numbers in §0 only as a sanity check.
**Done-when:** `results/baseline/*.json` + table in `docs/baseline_results.md`.

### Phase 8 — Failure-case & efficiency analysis
- `scripts/analyze_failures.py`: per-image mIoU ranking; worst-k gallery (image | GT | pred | error map | invalid mask); normalized per-class confusion matrix; IoU vs. image brightness (mean luma bins), contrast (std of luma), saturated-pixel ratio (glare), object size (segment-area bins), boundary-band accuracy (trimap ±k px around GT edges); **valid vs invalid region performance** using `invIds`; seen-drive vs unseen-drive gap.
- Categorize failures as the supervisor listed: low illumination, glare/over-exposure, shadows, low contrast, small objects (poles, traffic lights/signs, persons, riders). Quantify, don't just show pictures.
- Efficiency limits: where is latency/FLOPs concentrated (per-stage profile)?
- Write `docs/failure_analysis.md` ending with a ranked list of **3–5 concrete limitations with evidence**.
**Done-when:** each limitation backed by a figure or number.

### Phase 9 — Literature review & research gap
- Focused review (target 15–25 papers), recorded in `docs/literature_review.md` as a table: *paper, year, venue, idea, dataset, extra compute?, limitation*. Suggested clusters (verify each paper exists and read the abstract/method before citing):
  1. **ACDC & adverse-condition segmentation:** ACDC (Sakaridis et al., ICCV 2021, arXiv 2104.13395), its benchmark leaderboard (night supervised results), the Dark Zurich/GCMA/MGCDA lineage (ICCV 2019 / TPAMI), DANNet, DANIA, CCDistill, other night-time adaptation work
  2. NightCity & night segmentation: NightCity paper (arXiv 2003.06883), illumination-/exposure-aware approaches
  3. Low-light enhancement as preprocessing vs. joint learning (Retinex-based, Zero-DCE family) and why separate enhancement can hurt segmentation
  4. Lightweight/efficient segmentation (BiSeNet, SegFormer-B0, PIDNet, DDRNet, etc.) as efficiency reference points
  5. State-space-model segmentation: VMamba, VM-UNet, Mamba-UNet, and recent SSM-segmentation variants
  6. Boundary-aware / small-object segmentation; frequency-domain and illumination-invariant feature learning
- Use search to find **recent (2023–2026)** work; confirm titles/venues before listing.
- `docs/research_gap.md`: 1 page — *what existing methods do, what none of them handle, why that matters for lightweight nighttime segmentation, and a falsifiable hypothesis* ("Doing X reduces failure mode Y by ≥Z mIoU at <W% extra FLOPs").
**Done-when:** gap statement and hypothesis agreed with owner.

### Phase 10 — Novel method design (owner decides; IDE proposes with evidence)
The supervisor warns: **no plain "add attention module" solutions.** Each candidate must have a technical rationale tied to a measured failure from Phase 8. Candidate directions (examples, not prescriptions):
1. **Illumination-invariant feature decomposition inside the encoder** — a lightweight illumination/reflectance-style split (Retinex-inspired) at the stem, feeding reflectance-like features to the SSM blocks; trained end-to-end with the segmentation loss only (no separate enhancement network at inference).
2. **Illumination-adaptive scan/gating in SS2D** — condition the state-space parameters (Δ, B, C) or scan ordering on a cheap global illumination descriptor, so dark/glare regions get different state dynamics.
3. **Boundary/small-object preservation** — high-resolution shallow-feature path + boundary-aware auxiliary loss (distance-transform or edge head, training-time only → zero inference cost).
4. **Training-time-only night↔day reference consistency** — ACDC gives a GPS-matched normal-condition `*_ref` image for every night image (400 train pairs). Use it only during training (feature-consistency or distillation from a day-branch/teacher) so that the night branch learns day-like features; inference stays single-image with **zero extra cost**. Risks to test explicitly: pairs are **not pixel-aligned** (need coarse/region-level alignment or global/class-level statistics, not per-pixel matching), dynamic objects differ, and reference images are unlabeled in this download (pseudo-labels from a day-trained model are noisy).
5. **Train-time-only illumination augmentation / synthesis** (physically motivated night simulation: gamma, noise model, glare blobs) as a generalization lever with no inference overhead — evaluate especially on the val-unseen-drive subset.
- For each candidate produce a 1-page card in `docs/method.md`: motivation (which measured failure), mechanism, expected effect, extra params/FLOPs estimate, risks, minimal experiment to falsify. Rank by (expected gain × feasibility ÷ compute cost). Implement only the top choice first; keep others as backup.
**Done-when:** one method chosen with a written hypothesis and an experiment plan.

### Phase 11 — Implementation of the proposed method
- Implement in new files (`src/models/novel_*.py`, extra loss in `losses.py` behind config flags). Baseline path must stay bit-identical when flags are off (test it).
- Keep parameter/FLOP overhead small and measured; the "lightweight" claim must be *demonstrated*, not stated.
- Unit tests for new modules (shape, gradient flow, determinism); smoke-overfit test again.

### Phase 12 — Experiments & ablations
- **Main comparison:** baseline vs. proposed under the *same* protocol, seeds (≥3), splits, schedule, resolution (Rules 25/28). Table with all metrics (val overall / seen / unseen) + params/FLOPs/latency/FPS.
- **Ablations** (one factor at a time, in `docs/ablations.md`): each proposed component on/off; loss-term weights; module placement; resolution/efficiency trade-off; train-only vs. inference-time components; sensitivity to training-set size (e.g. 100 / 200 / 400 images).
- **External comparison** where feasible: a small standard network trained under the same protocol, and/or published ACDC night numbers (cite precisely, state differences in architecture/resolution/protocol; don't compare apples to oranges silently).
- **Statistical care:** mean ± std over seeds; state when differences are within noise.
- Re-run failure analysis on the new model: show *which failure categories improved* — this is the evidence for the hypothesis.
- If the owner approves: submit the final model's test predictions to the server once.
- Negative/neutral outcomes: report them and discuss honestly.

### Phase 13 — Report & submission package
- `report/` (LaTeX or Markdown→PDF), common structure shared with Kanav:
  1. Introduction & problem 2. Related work 3. Dataset & protocol (ACDC night, splits, leakage notes, license) 4. Baseline implementation & results 5. Failure analysis 6. Research gap & hypothesis 7. Proposed method 8. Experiments, comparisons, ablations 9. Efficiency analysis 10. Limitations & future work 11. Reproducibility statement (commit hash, configs, hardware) 12. References
- Final checklist vs. the deliverables list: `model.py`, `train.py`, `inference.py`, dataset config/preprocessing, logs + checkpoints, quantitative results + sample predictions, failure analysis, literature review + gap, novel method code, comparisons + ablations, report.
- Final `README.md` reproduction commands tested from a clean clone (no dataset files in the repo).

---

## 4. Suggested timeline (adjust to deadlines; today is 5 Nov 2026)

| Window | Phases |
|---|---|
| Week 1 | 0, 1, 2 (+ send protocol questions to supervisor; owner uploads night-only private Kaggle Dataset) |
| Week 2 | 3, 4, 5, 6 |
| Weeks 3–4 | 7 (long runs in background while doing 9), 8 |
| Week 4–5 | 9, 10 |
| Weeks 5–7 | 11, 12 |
| Week 8 | 13, buffer |

300-epoch runs can take long on large inputs — benchmark one epoch first and estimate total time (× seeds) before launching.

---

## 5. Working protocol for the IDE assistant (per task)

1. State which **phase/task** you are on and re-read the relevant rules.
2. Inspect existing files before writing; reuse, don't duplicate.
3. Make the smallest change that works; show a diff summary.
4. Run tests/lint; paste the actual output; fix failures.
5. Update docs (`PROGRESS.md`, `EXPERIMENT_LOG.md`, `DEVIATIONS.md` as relevant) and commit.
6. End each task with: **what was done · what was verified · what is uncertain · what needs the owner's decision.**
7. When blocked or when a result looks too good/bad: investigate the cause (leakage? ignore index? wrong label map? eval on train data?) before reporting it.

### Common pitfalls to actively check
- Label ID vs trainId confusion → silent wrong mIoU. Use `*_gt_labelTrainIds.png` only; check unique values in every loaded mask.
- Confusing the **`*_ref` folders** (day reference images, no labels) with the night images, or loading `*_rgb_ref_anon.png` as training input by mistake.
- Double-nested paths (`gt/gt/...`, `rgb_anon/rgb_anon/...`) and the filename change `_rgb_anon.png` → `_gt_labelTrainIds.png`.
- **Same-drive leakage:** GOPR0351 is in both train and val. Always report the unseen-drive subset; never create your own random re-splits.
- Test split has **no labels** — any metric computed on it is a bug.
- Forgetting `ignore_index=255` in loss **and** metrics.
- Averaging per-image IoU instead of dataset-level confusion-matrix IoU.
- Resizing masks with bilinear interpolation.
- FLOPs missing selective-scan ops; FPS measured without CUDA sync or warm-up.
- Evaluating with different resolutions for baseline vs. proposed method.
- BatchNorm/LayerNorm behaviour with batch size 1–2 under large inputs.
- Mamba CUDA-kernel version mismatches silently falling back to slow paths.
- Committing dataset files or predictions overlaid on dataset images to a public repo (license).

---

## Appendix A — Common results schema (share with Kanav)

`results.json` per run:
```json
{
  "run_id": "", "dataset": "acdc_night|nightcity", "protocol": "official_train_val", "model": "baseline|novel_x",
  "git_commit": "", "seed": 0, "config": "configs/....yaml",
  "eval_split": "val|val_seen|val_unseen", "checkpoint": "best|last",
  "input_resolution": [H, W], "eval_resolution": [H, W],
  "optimizer": "", "lr": 0.0, "epochs": 0, "loss": "",
  "metrics": {"mIoU": 0.0, "per_class_IoU": {"road": 0.0}, "dice_mean": 0.0, "per_class_dice": {},
              "mPA": 0.0, "pixel_acc": 0.0, "sensitivity_macro": 0.0, "specificity_macro": 0.0},
  "efficiency": {"params_M": 0.0, "flops_G": 0.0, "flops_method": "", "latency_ms": 0.0, "fps": 0.0,
                 "batch_size": 1, "precision": "fp32", "hardware": "GPU/driver/CUDA/PyTorch"},
  "notes": ""
}
```
Tables: (1) baseline metrics, (2) per-class IoU, (3) efficiency, (4) baseline vs. proposed, (5) ablations — same column order for both datasets.

## Appendix B — Immediate next actions for the owner
1. Download `model.py` from the supervisor's Drive link into `src/models/`.
2. Read `License.pdf` and both `README.md` files in the ACDC download; create the **private** night-only Kaggle Dataset (§0.5 item 1).
3. Check the ACDC website for the test-server submission format and quota; paste what it says into `docs/protocol.md`.
4. Send the four protocol questions in Phase 3 to the supervisor.
5. Point the IDE at this file and start **Phase 0 → 2**; share `docs/dataset_audit.md` back before Phase 3 sign-off.
6. Confirm with Kanav that you both use Appendix A.

## Appendix C — Change log

**v3 — 5 Nov 2026 (dataset changed Dark Zurich → ACDC night, after the supervisor's decision and the ACDC directory tree)**
- Dataset facts replaced by the verified ACDC night table: 400 train / 106 val labeled, 500 test withheld; reference day images 1-to-1 for every image; invalid masks available. Fog/rain/snow out of scope.
- Protocol simplified: supervised training on the official train split, selection on val; the earlier blocked k-fold on 50 frames, source-only/UDA options and Cityscapes download are removed.
- New leakage handling: GOPR0351 is in train and val → report val-seen-drive (56) and val-unseen-drive (GOPR0356, 50) separately; always also report last-epoch results.
- Test-set handling added: predictions only; ACDC server submission only with owner approval; export script; format/quota to be confirmed by the owner (unverified).
- Dark Zurich removed from the plan; its val/test frames coincide with ACDC night val/test frames (by sequence+index), so it is not an independent test.
- Loader changed to `acdc.py` with the double-nested layout and `_rgb_anon` → `_gt_labelTrainIds` pairing; optional reference-image mode (training-only).
- Phase 8 now uses the invalid masks; Phase 10 candidate 4 rewritten around the reference day images; ≥3 seeds now feasible; ablation on training-set size kept.
- Kaggle: private night-only dataset upload added; license note added; compute rules otherwise unchanged.

**v2 — 4 Nov 2026 (Dark Zurich tree + Kaggle compute)** — superseded by v3 except the Kaggle compute rules, which carry over.
