# Changelog

All notable changes, grouped by phase.

## [Unreleased]

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
