# P004 — Config & run system

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Every experiment is driven by a configuration file with a stable hash, is seeded, and writes into its own run folder.

## Inputs / dependencies
P003 environment.

## Work log
1. `stratia/utils/config.py`: YAML configs composed with OmegaConf (`defaults:` lists relative to the file, then the file, then `key=value` overrides); `config_hash()` = SHA-256 of the resolved config with sorted keys.
2. `stratia/utils/seed.py`: seeds Python, NumPy and PyTorch (CPU+CUDA); deterministic mode disables cuDNN autotuning, requests deterministic kernels (warn-only) and sets `CUBLAS_WORKSPACE_CONFIG`; DataLoader worker seeding.
3. `stratia/utils/run.py`: `Run` context manager creating `runs/<UTC>_<name>_<cfghash>/` with `config.yaml`, `env.json` (git SHA and dirty flag, Python/torch/CUDA, GPU, lock-file hash) — metrics and registry are P005.
4. `scripts/repro_check.py` + `configs/repro_check.yaml`: a small GPU training job (MLP on seeded synthetic data, AdamW, fp16 autocast, 3 epochs) run twice with one config.
Hydra is installed but only OmegaConf composition is used; Hydra's application framework is not needed yet.

## Verification
`python scripts/repro_check.py` → `"identical": true`, exit code 0. Both runs (config hash `b011a8395a`) produced the same per-epoch values:

| Epoch | Loss | Accuracy |
|---|---|---|
| 0 | 1.6448174285888673 | 0.5688 |
| 1 | 0.6841289682388305 | 0.86945 |
| 2 | 0.4003426077842712 | 0.91775 |

## Exit criteria
- [x] Two runs with the same config produce identical first-epoch (and all-epoch) metrics.

## Fit & data-risk notes
Determinism is verified empirically for this job; some CUDA kernels used by larger models may still be non-deterministic, so headline results will be reported as mean ± std over seeds.

## Deviations from plan & why
None.

## Next phase
P005 — Experiment tracking.
