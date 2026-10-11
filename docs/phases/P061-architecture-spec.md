# P061 — Architecture spec

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Freeze the v1 model design before any head is written: the diagram, every tensor shape from input to contract
output, the mapping of each contract output to a module and a loss, the CBH bin edges, and the two decisions the
plan asks for as ADRs (heads, losses).

## Inputs / dependencies
`docs/contract.md` v1.0 and `stratia/contract.py` (ADR-001); ADR-002 (DINOv3 ViT-S/16); ADR-004 (native view, ray
maps); P010 compute profile (batch 16 at 512 px, no BatchNorm); P050 null-token encoders; P054/P057/P058 baseline
numbers (what the heads have to beat); configs/ontology.yaml (genus → étage, étage height thresholds).

## Work log
1. `docs/architecture.md`: diagram; shape table (image 512 → 1,024 tokens of 384, heads at [N,12] / [N,3] / [N,9] /
   [N,4,128,128] / [N,3,128,128] / [N,25] / [N,1]); parameter budget ≈ 1.5 M trainable with the backbone frozen;
   CBH bins; contract → module → loss → supervision table; module map with the phase that writes each file; the
   frozen-feature training path (cls for global heads, the 32 × 32 grid for dense heads, an 8 × 8 pooled grid for
   the CBH head).
2. `docs/adr/ADR-005-heads.md`: attention pool ⊕ CLS for the global heads, CORAL for oktas, a light DPT-style
   decoder shared by sky parsing and layers (GroupNorm, two fusion blocks to H/4), a CBH head with zenith-prior
   attention over tokens and a 24-bin + no-cloud distribution, energy + Mahalanobis for the OOD score; no
   BatchNorm; branch-free for ONNX.
3. `docs/adr/ADR-006-losses.md`: set-valued genus BCE with partial negatives (weight 0.3 for single-label datasets),
   soft BCE on expert shares, CORAL with soft cumulative targets, hierarchy consistency (noisy-OR genus → étage),
   dense CE with ignore, CBH = CE on a log-height Gaussian target + CRPS, layer consistency = KL from the
   CBH-implied lowest étage to the étage head (gradient to CBH only), Kendall weighting for the sum.
4. Bin design: 24 bins log-spaced inside each étage with the étage thresholds (2,000 and 6,000 m) as edges, so the
   consistency term is a sum over bins; bin 24 = no cloud overhead.

## Verification
- Review against the contract: every input and output name, shape and class order in `docs/architecture.md` §2
  equals `stratia/contract.py` (`output_shapes(n, 512, 512, k_cbh=24)`); the P071 contract tests will hold the
  implementation to it.
- Review against the compute profile: all heads sum to ≈ 1.5 M parameters and have no BatchNorm; the dense
  decoder's largest activation is [16, 64, 128, 128] fp16 = 32 MB per batch of 16.
- Review against the data: every supervision source in §4 exists in `data/labels.parquet`,
  `data/montenegro_targets.parquet`, the P018 masks or `data/cbh_pairs.parquet`; the sun-glare class has no
  labels yet and is listed as supervised by the augmentation's glare mask (P074).

## Exit criteria
- [x] Reviewed (this doc; the owner's review is the stage push).

## Fit & data-risk notes
- Partial-negative weight, CBH target width σ, λ_crps and λ_layer are the loss hyperparameters with no prior
  evidence; each is a config field and P080 ablates them.
- The 8 × 8 CBH token grid (64 cells of 64 px) is a Stage G–H compromise for disk; the head itself is grid-agnostic
  and runs on the 32 × 32 grid when the backbone is in the graph.

## Deviations from plan & why
- None in substance. The plan's "ordinal bins" are specified as étage-aligned log bins rather than uniform
  log-spacing over 0–12 km, so the layer-consistency term needs no fractional bin membership.

## Next phase
P062 — SkyRay encoder.
