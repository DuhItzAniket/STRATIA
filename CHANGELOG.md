# Changelog

All notable changes, grouped by phase.

## [Unreleased]

### P003 — Project environment (2026-10-04)
- Clean project virtual environment; `requirements.in` (top level) and `requirements.lock` (97 pinned packages, PyTorch 2.5.1 + CUDA 12.1).
- `python -m stratia.env_check`: versions, GPU/driver, CUDA matmul accuracy, fp16/bf16 autocast, NumPy < 2 guard, gated DINOv3 access — all passing.

### P002 — Phase protocol & logs (2026-10-04)
- Added `docs/PLAN.md` (100-phase plan with 2026-10-04 corrections: exact DINOv3 model IDs, Bengaluru airports VOBL/VOBG for METAR, HDD archive + SSD cache storage layout).
- Added phase template, `CONTRIBUTING.md` (phase workflow and research honesty rules), `PROJECT_STATE.md`, this changelog and `EXPERIMENT_LOG.md`.

### P001 — Repo bootstrap (2026-10-04)
- README, Apache-2.0 licence, ignore rules, line-ending rules, `stratia` package skeleton.
