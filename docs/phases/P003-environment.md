# P003 — Project environment

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
A reproducible, project-local Python environment with pinned versions and an automated environment check.

## Inputs / dependencies
P001–P002. NVIDIA driver 566.24 (CUDA 12.7 capable).

## Work log
1. Created a clean virtual environment `.venv` (Python 3.11.9), **not** sharing the global site-packages: the global environment also holds TensorFlow and other packages that could change results silently.
2. `requirements.in` lists top-level dependencies (PyTorch 2.5.1 + CUDA 12.1 wheels, transformers 5.5.4, NumPy < 2, data, geometry and experiment tools); `requirements.lock` freezes all 97 installed packages (with the PyTorch wheel index on its first line).
3. `stratia/env_check.py` (`python -m stratia.env_check [--json PATH] [--skip-hf]`): package versions, `nvidia-smi` query, CUDA matmul accuracy and throughput, fp16/bf16 autocast, NumPy < 2 guard, and gated DINOv3 access (config files only). Non-zero exit on failure.

## Verification
`python -m stratia.env_check` → `ok: true`, no failures:

| Check | Result |
|---|---|
| GPU / driver | NVIDIA GeForce RTX 4050 Laptop GPU, driver 566.24, 6,141 MiB |
| PyTorch / CUDA / cuDNN | 2.5.1+cu121 / 12.1 / 9.1.0; bf16 supported |
| FP32 matmul 2048² | 4.6 TFLOP/s, max relative error 2.1e-6 vs float64 |
| Autocast fp16 / bf16 | finite outputs |
| Key packages | torch 2.5.1+cu121, torchvision 0.20.1, transformers 5.5.4, numpy 1.26.4, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1, xarray 2026.9.0, netCDF4 1.7.4, h5py 3.16.0, zarr 3.1.5, Pillow 12.3.0, OpenCV 4.11.0, imagehash 4.3.2, pvlib 0.16.1, omegaconf 2.3.1, peft 0.21.2, timm 1.0.30 |
| Hugging Face | user `DuhItzAniket`; DINOv3 ViT-S/16, ViT-B/16, ViT-L/16: ok |

## Exit criteria
- [x] `env_check` prints GPU, driver, CUDA and versions.
- [x] CUDA matmul and AMP tests pass.

## Fit & data-risk notes
NumPy is pinned below 2 (the CloudScope prototype broke when NumPy 2 replaced 1.x under an older PyTorch).

## Deviations from plan & why
- The JSON report is not committed (it contains local file paths); its content is summarised above.
- `pandas` resolved to 3.0.6; code must avoid behaviour removed in pandas 3 (copy-on-write is the default).

## Next phase
P004 — Config & run system.
