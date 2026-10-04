# P010 — Hardware budget profiling

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Measure, not estimate, what the 6 GB RTX 4050 can do with DINOv3 ViT-S/B/L at 224/384/512 px: inference, full fine-tuning (with and without gradient checkpointing) and LoRA; then fix default batch sizes and resolutions.

## Inputs / dependencies
P003 environment; gated DINOv3 weights (downloaded: ViT-S 86 MB, ViT-B 343 MB, ViT-L 1.2 GB).

## Work log
1. `scripts/profile_backbones.py`: per backbone and resolution, bf16 inference at batch 32; fp16 AdamW training with a power-of-two batch search (full, full + checkpointing, LoRA r=8 on q/k/v); results saved after each configuration.
2. **First attempt stalled** (GPU memory full, 0% utilisation, 15 minutes on ViT-S): the Windows driver's CUDA sysmem fallback spilled into system RAM instead of raising out-of-memory. Fixed by capping the allocator (`set_per_process_memory_fraction`, 0.90) and rejecting batches whose throughput collapses below 30% of the previous one. A ViT-S@224 test then completed in 32 s.
3. Full run: all 9 configurations completed.
4. Wrote `docs/compute.md` (measurements, spill countermeasures, decisions, cache sizes).

## Verification
Selected results (full table in `docs/compute.md`, raw in `docs/compute_profile.json`):

| Setting | Result |
|---|---|
| ViT-S @512 inference | 122 img/s, 0.68 GB |
| ViT-S @512 full fine-tune | batch 16, 39 img/s, 3.93 GB |
| ViT-S @512 LoRA | batch 16, 40 img/s, 2.64 GB |
| ViT-B @512 + checkpointing | batch 32, 12.5 img/s, 3.76 GB |
| ViT-L @512 inference | 17 img/s, 3.07 GB |
| ViT-L full fine-tune + checkpointing | does not fit at batch 1 |

Free memory at start was 5.32 of 6.44 GB; two settings peaked above it (ViT-B@384 full, ViT-L@384 LoRA) and are flagged.

## Exit criteria
- [x] Table in `docs/compute.md`; default batch sizes chosen (ViT-S @512, batch 16, or 64 with checkpointing).

## Fit & data-risk notes
Batch 16 at 512 px is small. That is fine for the ViT backbone (it uses LayerNorm), but STRATIA's heads must avoid BatchNorm (use LayerNorm/GroupNorm), and gradient accumulation or checkpointing (batch 64) is available when larger effective batches help.

## Deviations from plan & why
- ViT-L full fine-tuning was not attempted without checkpointing (AdamW states alone exceed memory); with checkpointing it does not fit at batch 1.
- Profiling numbers are recorded in `docs/compute_profile.json` rather than the run registry (they are measurements of hardware, not experiments).

## Next phase
P011 — Inventory local data (Stage B).
