# P051 — Feature cache

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Cache frozen-backbone features once, keyed by backbone, input size and transform version, with a fingerprint
that ties every cache to its row list and preprocessing, so that the baseline stage and the frozen-feature training
stages read embeddings in seconds instead of re-running backbones.

## Inputs / dependencies
P022 image cache (768 px); P021 manifest (52,032 rows); P047 pairs (the 10-minute daytime frames, 20,942, added to
the image cache by `scripts/cache_cbh_frames.py`); P010 compute profile; DINOv3 weights (gated, the user's token);
P018 segmentation index (the dense subset).

## Work log
1. `stratia/features/backbones.py`: one interface over DINOv3 ViT-S/B/L (transformers `AutoModel`, CLS after the
   final norm = `cls`, registers and CLS stripped from the patch grid), DINOv2 S/B, CLIP ViT-B/16 (projected
   embedding = `cls`, 768-wide tokens) and SigLIP 2 B/16 (attention-pooled `cls`), each with its own normalisation
   and a text tower for the two contrastive models (P055). Full-frame resize to the input size (contract rule).
2. `stratia/features/cache.py`: `FeatureCache(cache_root, backbone, size, transform)` → `meta.json`, `index.parquet`,
   `cls.npy`, `mean.npy` (fp16 memmaps), optional `patches.npy` (N × tokens × width); fingerprint = SHA-256 of
   the ordered sample ids + model id + size + transform; resumable builds (`progress.json`); a build of another
   fingerprint in the same folder is refused; `features(sample_ids)` returns float32 rows in the requested order.
3. `scripts/build_feature_cache.py` (rows = manifest + CBH frames; `--patches seg` for the dense subset) and
   `scripts/cache_cbh_frames.py` (20,942 frames into the image cache at 60 frames/s).
4. Two failures fixed on the way: **DINOv3 ViT-L overflows in fp16** (every CLS row NaN; it now runs in bfloat16
   per `BackboneSpec.compute_dtype`, and the build refuses non-finite batches), and **OpenAI CLIP ships pickle
   weights only**, which transformers 5 refuses on torch < 2.6; `converted_checkpoint` turns them into safetensors
   once under `<cache_root>/hf_converted/`. CLIP's `mean` width (768) differs from its `cls` width (512); the cache
   stores both widths.

## Verification
- `tests/test_feature_cache.py` (4): fingerprint sensitivity, build / load / order / refusal, resume from a partial
  build, registry and grid helpers (a deterministic fake backbone; the real models are not loaded in CI).
- Built caches (`docs/data/feature_cache.md`), 72,974 rows each unless stated:

| Cache | Width | Disk | Build |
|---|---|---|---|
| dinov3_vits16 @ 512 | 384 | 0.10 GB | 522 s |
| dinov3_vits16 @ 512, patch grid, 8,870 dense images | 384 × 1,024 tokens | 6.51 GB | 78 s |
| dinov3_vits16 / vitb16 / vitl16 @ 224 | 384 / 768 / 1,024 | 0.10 / 0.21 / 0.28 GB | 129 / 234 / 656 s |
| dinov2_small / dinov2_base @ 224 | 384 / 768 | 0.10 / 0.21 GB | 114 / 221 s |
| clip_vitb16 @ 224 | 512 (tokens 768) | 0.17 GB | 206 s |
| siglip2_b16 @ 224 | 768 | 0.21 GB | 178 s |

  Total 7.90 GB; every cache finite (checked on the rebuilt ViT-L, CLIP and SigLIP 2). Throughput was bounded by
  JPEG decoding from the 768 px cache (~140 img/s at 512 px, 6 workers), not by the backbones.
- Reproducibility: the fingerprint of every cache is in the table; rebuilding with the same rows, model and size
  reproduces it (the no-op path is tested).

## Exit criteria
- [x] Cache reproducible (fingerprinted, resumable, refuses mismatches).
- [x] Disk budget logged (7.90 GB; the ViT-S patch grid for all 73k rows would be 58 GB and is not built).

## Fit & data-risk notes
- A stale cache after a transform change is the plan's guard: the transform version is part of the key and the
  fingerprint; `ffr1` is the plain full-frame resize, no mask zeroing (the contract's mask zeroing belongs to the
  final model's input and will be a new transform version when the loader applies it).
- fp16 storage holds values up to 65,504; the largest magnitude seen is 10.7 (CLIP). Compute precision is the
  backbone's concern (bfloat16 for ViT-L).

## Deviations from plan & why
- The plan says "fp16 memmap/zarr"; memmapped `.npy` files were enough and keep the dependency list unchanged.
- The patch grid is cached for the dense subset only (budget).

## Next phase
P052 — Legacy CNN.
