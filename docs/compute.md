# Compute budget — RTX 4050 Laptop GPU (6 GB)

Measured in P010 on 2026-10-04 with `scripts/profile_backbones.py` (raw numbers: [`compute_profile.json`](compute_profile.json)). PyTorch 2.5.1 + CUDA 12.1, Windows 11 (WDDM driver 566.24). Synthetic inputs; throughput excludes data loading.

## 1. Measurements

Inference: bf16 autocast, batch 32. Training: fp16 autocast + AdamW on all trainable parameters; "max batch" is the largest power of two (≤ 64) that fit under the allocator cap; images/s at that batch. Gradient checkpointing (ckpt) trades speed for memory; LoRA = rank 8 on q/k/v projections with the backbone frozen.

| Backbone @ input | Inference img/s (GB) | Full fine-tune: batch, img/s (GB) | Full + checkpointing | LoRA |
|---|---|---|---|---|
| ViT-S/16 @224 | 751 (0.24) | 64, 237 (3.15) | 64, 185 (0.84) | 64, 253 (2.09) |
| ViT-S/16 @384 | 226 (0.45) | 32, 74 (4.44) | 64, 58 (1.89) | 32, 77 (2.98) |
| **ViT-S/16 @512** | **122 (0.68)** | **16, 39 (3.93)** | **64, 30 (3.12)** | **16, 40 (2.64)** |
| ViT-B/16 @224 | 243 (0.73) | 32, 87 (4.29) | 64, 69 (2.38) | 64, 102 (4.39) |
| ViT-B/16 @384 | 92 (1.09) | 16, 29 (5.58) ⚠ | 64, 24 (4.10) | 16, 34 (3.35) |
| ViT-B/16 @512 | 50 (1.50) | 4, 15 (3.36) | 32, 12.5 (3.76) | 8, 18 (3.01) |
| ViT-L/16 @224 | 94 (2.08) | not tried | does not fit (batch 1) | 16, 41 (4.34) |
| ViT-L/16 @384 | 32 (2.52) | not tried | does not fit | 8, 12 (5.46) ⚠ |
| ViT-L/16 @512 | 17 (3.07) | not tried | does not fit | 4, 6.5 (4.99) |

⚠ Peak above the 5.32 GB that was free at start (Windows and other applications hold ≈ 1.1 GB): the run may have used a little shared system memory. Use the next smaller batch for these settings.

## 2. Windows memory spill (important)

The first profiling attempt stalled: GPU memory full, 0% utilisation, still on ViT-S after 15 minutes. Windows NVIDIA drivers with **CUDA sysmem fallback** move allocations into system RAM instead of raising out-of-memory, so an oversized batch silently runs orders of magnitude slower. Countermeasures:
1. All STRATIA training scripts cap the allocator with `torch.cuda.set_per_process_memory_fraction(...)` at about the memory free at start (≈ 0.80 of 6.44 GB), so oversized batches fail fast.
2. Recommended for the owner: NVIDIA Control Panel → Manage 3D settings → Program settings → `python.exe` (in `.venv`) → **CUDA – Sysmem Fallback Policy = Prefer No Sysmem Fallback**.
3. Close GPU-using applications (browsers with hardware acceleration, ChatGPT desktop) during long runs.

## 3. Decisions for STRATIA

| Use | Setting | Why |
|---|---|---|
| Feature caching (frozen) | ViT-S @512: ≈ 122 img/s → 100k frames ≈ 14 min; ViT-L @512: ≈ 17 img/s → 100k ≈ 1.6 h | Frozen-feature stages (P051, P073–P076) are cheap |
| Main training (P077) | ViT-S @512, LoRA or partial fine-tune, batch 16 (or 64 with checkpointing), ≈ 30–40 img/s | ≈ 12–17 min per epoch for 30k images |
| Capacity ablation | ViT-B @512 with checkpointing, batch 32, ≈ 12.5 img/s | ≈ 40 min per 30k-image epoch; use sparingly |
| ViT-L | Frozen features only; LoRA possible but slow (6.5 img/s @512) | Baselines and feature studies (P054) |
| 7B | Not usable (26.9 GB of weights) | ADR-002 |

## 4. Storage per cached frame (fp16 patch tokens, 32 × 32 grid at 512 px)

| Backbone | Width | MB per frame | 100k frames |
|---|---|---|---|
| ViT-S/16 | 384 | 0.79 | 79 GB |
| ViT-B/16 | 768 | 1.57 | 157 GB |
| ViT-L/16 | 1024 | 2.10 | 210 GB |

Full patch grids are cached only for labelled or dense-task subsets; pooled tokens (CLS + pooled patches, a few KB per frame) are cached for everything else. Caches live on the internal SSD (`cache_root`).
