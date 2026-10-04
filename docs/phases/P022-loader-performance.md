# P022 — Loader performance

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Feed the GPU fast enough: a resized image cache on the internal SSD and a measured DataLoader configuration for Windows.

## Inputs / dependencies
P021 manifest; P010 compute budget (training needs ≈ 30–120 img/s).

## Work log
1. `stratia/data/image_cache.py`: cache layout `<cache_root>/img768/<image_file>.jpg` (longest side ≤ 768 px, aspect preserved, JPEG quality 95, never upscaled); OpenCV I/O through `np.fromfile`/`imdecode` (Windows paths with spaces); `CachedImageDataset` returning 512 × 512 uint8 CHW tensors (full frame, no crop, as in stratia-contract v1).
2. `scripts/build_image_cache.py`: process pool, skip existing files, error report.
3. `scripts/bench_loader.py`: images/s to the GPU (pinned memory, non-blocking copies, persistent workers, prefetch 4), cache vs originals, 0/4/8 workers; results in `docs/data/loader_benchmark.json`.
4. Tests: longest-side resize with aspect ratio, PNG source and a path with a space, no upscaling, RGB channel order, dataset tensor shape and type.

## Verification
- Cache: **52,032 images, 0 errors, 3.16 GB**, built in 258 s (≈ 200 img/s, 12 processes).
- Throughput to the GPU, 512 × 512, batch 32, Eye2Sky sample:

| DataLoader workers | Cache (img/s) | Original 2112 × 2048 JPEGs (img/s) |
|---|---|---|
| 0 | 89.8 | 33.6 |
| 4 | 587.1 | 125.2 |
| 8 | **816.7** | 247.4 |

## Exit criteria
- [x] ≥ 300 img/s to the GPU at 512 px (816.7 with 8 workers; 587.1 with 4).

## Fit & data-risk notes
- With caching, data loading is 7–20× faster than the training/inference rates measured in P010, so it will not bottleneck training.
- The cache re-encodes PNG sources (SWIM family) as JPEG quality 95: negligible for training images; masks are never cached this way.

## Deviations from plan & why
- Cache resolution 768 px (longest side) instead of 512: keeps headroom for crop/scale augmentation (P048) before the final 512 × 512 input; the cost is 3.16 GB in total.

## Next phase
P023 — Integrity check (Stage C).
