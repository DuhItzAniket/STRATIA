# Changelog

All notable changes, grouped by phase.

## [Unreleased]

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
