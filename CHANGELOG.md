# Changelog

All notable changes, grouped by phase.

## [Unreleased]

### P018 — Segmentation datasets loader (2026-10-04)
- Segmentation index and unified label mapping for SWIMSEG, SWINSEG, SWINySEG, SHWIMSEG and Almería (`stratia/data/segmentation.py`); encodings verified from the data (JPEG masks thresholded), QA sheets per dataset.

### P017 — MGCD parser + weather (2026-10-04)
- MGCD reader with official split, classes and weather joined for all 8,000 images (`stratia/data/mgcd.py`).

### P016 — Montenegro parser (2026-10-04)
- Montenegro parser and per-image soft labels (`stratia/data/montenegro.py`), all README counts reproduced, archive image-name prefix handled; experts' majority share on cloud-base height only 0.58.

### P015 — Acquire missing public data (2026-10-04, PARTIAL)
- Verified licences from official records (CCSN CC0, Almería and Montenegro CC BY 4.0, Eye2Sky CDLA-Sharing 1.0, LenghuSky-8 Apache-2.0); found that DeepSky images are only available on request (Zenodo holds the paper only) and WEBCAM has no public link; owner actions in `docs/data/acquisition.md`.

### P014 — Ceilometer reader (2026-10-04)
- CHM15k ceilometer reader with QC flags (`stratia/data/ceilometer.py`), summary over all 236 days and time-height figures; matches the vendor plot; CDLRB file latitude error and CDLRA 0x8000 laser-ageing warning identified and handled.

### P013 — Eye2Sky readers (2026-10-04)
- Eye2Sky readers (`stratia/data/eye2sky.py`) and the OCamCalib fisheye model (`stratia/geometry/ocam.py`); axis convention measured on 38 calibrations (xc = row), mask-name mismatch resolved, round trip ≈ 4e-7 px on OLDLR, WESTE and AURIC.

### P012 — Eye2Sky inventory (2026-10-04)
- `scripts/eye2sky_inventory.py` and `docs/data/eye2sky_inventory.md`: images per station/day, calibration validity windows, ceilometer coverage (118 days each), stations within 1 km of the ceilometers (OLDLR, OLUOL, OLWIN; WESTE); 0 pairable image-days on disk — escalated.

### P011 — Inventory local data (2026-10-04)
- `scripts/inventory.py`: per-file inventory of the data root (dataset, format, size, resolution, mode, junk, unreadable) and `docs/data/inventory.md`; 64,406 files, 15.0 GB, 0 unreadable images; CCSN mixed-resolution shortcut risk recorded.

### P010 — Hardware budget profiling (2026-10-04)
- Measured DINOv3 ViT-S/B/L memory and throughput at 224/384/512 px (inference, full fine-tune, checkpointing, LoRA); `docs/compute.md` with defaults (ViT-S @512, batch 16 or 64 with checkpointing) and Windows sysmem-fallback countermeasures (allocator cap, throughput-collapse guard).

### P009 — Research design & pre-registration (2026-10-04)
- Pre-registered research design (`docs/research_design.md`): hypotheses H1–H4, metrics, Gate G4 criteria C1–C3, locked-test policy, threats to validity; ADR-002 (DINOv3 ViT-S/16 backbone), ADR-003 (single-camera scope); paper outline.

### P008 — Interface contract v1 (2026-10-04)
- `stratia-contract` v1.0 (`docs/contract.md`), class-order constants (`stratia/contract.py`), model-card JSON Schema, ADR-001; Sun-aligned unit-vector ray maps and a 3-value Sun metadata input (no time/date/location shortcuts).

### P007 — Data registry spec (2026-10-04)
- Data registry `configs/datasets.yaml` (12 datasets with source, licence and verification status, paths, image globs, expected counts), machine paths template, registry loader/validator and `scripts/check_registry.py`; all present datasets match their expected image counts.

### P006 — Tests & CI (2026-10-04)
- Unit tests for config, seeding and run bookkeeping; ruff and pytest configuration; GitHub Actions CI (CPU-only PyTorch, lint, tests, gitleaks); pre-commit configuration.

### P005 — Experiment tracking (2026-10-04)
- Metrics log (JSONL + optional TensorBoard), run summary, and `runs/registry.csv` row per run (git SHA, dirty flag, config hash, data hash, status, metrics); robust when no GPU is visible.

### P004 — Config & run system (2026-10-04)
- Config composition with a stable config hash, global seeding/determinism, per-run folders with config and environment snapshot; reproducibility check: two GPU runs with one config gave bit-identical metrics.

### P003 — Project environment (2026-10-04)
- Clean project virtual environment; `requirements.in` (top level) and `requirements.lock` (97 pinned packages, PyTorch 2.5.1 + CUDA 12.1).
- `python -m stratia.env_check`: versions, GPU/driver, CUDA matmul accuracy, fp16/bf16 autocast, NumPy < 2 guard, gated DINOv3 access — all passing.

### P002 — Phase protocol & logs (2026-10-04)
- Added `docs/PLAN.md` (100-phase plan with 2026-10-04 corrections: exact DINOv3 model IDs, Bengaluru airports VOBL/VOBG for METAR, HDD archive + SSD cache storage layout).
- Added phase template, `CONTRIBUTING.md` (phase workflow and research honesty rules), `PROJECT_STATE.md`, this changelog and `EXPERIMENT_LOG.md`.

### P001 — Repo bootstrap (2026-10-04)
- README, Apache-2.0 licence, ignore rules, line-ending rules, `stratia` package skeleton.
