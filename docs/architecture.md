# STRATIA v1 architecture (P061)

Status: specified 2026-10-11 (Stage G); the contract it serves is `docs/contract.md` v1.0. Decisions: ADR-002
(backbone), ADR-004 (native view + ray maps), ADR-005 (heads), ADR-006 (losses).

## 1. Diagram

```
image [N,3,512,512] ──► DINOv3 ViT-S/16 (frozen in Stages G–H; LoRA in P077) ──► tokens T [N,1024,384], cls [N,384]
ray_map [N,4,32,32] ──► SkyRayEncoder ──► R [N,1024,384] ─────────────────────────────► T' = T + α·R
meta [N,3] ─────────► MetaFiLM ────────────────────────────────────────────────────────► cls' = FiLM(cls), T'' = FiLM(T')
                                                                                            │
     ┌───────────────────────────┬──────────────────────────────┬────────────────────────────┴──────┐
 GlobalHeads                 DenseDecoder (DPT-lite)        CbhHead                           OodScore
 AttentionPool(T'') ⊕ cls'   reassemble T'' → [N,384,32,32]  zenith-weighted attention pool  energy(genus_logits)
 → MLP → genus [N,12]        → 128 → 96 @64² → 64 @128²      over T'' with prior from        + Mahalanobis(pooled)
          étage [N,3]        → SkyParseHead 1×1 → [N,4,128,128]   ray_map z (cos zenith)     → ood_score [N,1]
          oktas CORAL [N,9]  → LayerHead    1×1 → [N,3,128,128]  ⊕ cls' → MLP → cbh [N,25]
```

Null geometry (all-zero `ray_map` / `meta`) is handled inside the encoders (P050): null tokens and null FiLM
parameters, so every head runs with or without calibration. No input carries time, date or location.

## 2. Tensor shapes (contract v1, input 512, patch 16)

| Stage | Tensor | Shape | Notes |
|---|---|---|---|
| input | `image` | [N, 3, 512, 512] | float32 in [0, 1] then normalised with the model card's mean/std; camera-mask pixels 0 |
| input | `ray_map` | [N, 4, 32, 32] | unit direction in the Sun-aligned frame + valid flag; zeros when unknown |
| input | `meta` | [N, 3] | [cos z☉, sin z☉, valid]; zeros when unknown |
| backbone | `tokens` | [N, 1024, 384] | patch tokens after the final norm, registers and CLS stripped |
| backbone | `cls` | [N, 384] | pooled output |
| geometry | `R` | [N, 1024, 384] | SkyRay tokens; null token on invalid patches; added with a learned scalar α (init 0.1) |
| geometry | `cls'`, `T''` | as above | FiLM from meta; identity when meta is null |
| global | `pooled` | [N, 384] | one-query multi-head attention pool over `T''` (4 heads, LayerNorm) |
| global | `genus_logits` | [N, 12] | sigmoid, multi-label |
| global | `etage_logits` | [N, 3] | sigmoid, multi-label |
| global | `oktas_probs` | [N, 9] | CORAL: 8 cumulative logits with shared weights and ordered biases → class probabilities (sum to 1) |
| dense | `feat` | [N, 64, 128, 128] | decoder output at H/4 |
| dense | `sky_parse_logits` | [N, 4, 128, 128] | invalid, sky, cloud, sun_glare |
| dense | `layer_logits` | [N, 3, 128, 128] | low, mid, high; read on cloud pixels only |
| cbh | `cbh_probs` | [N, 25] | 24 height bins + "no cloud overhead"; softmax over 25 |
| ood | `ood_score` | [N, 1] | higher = less trustworthy; threshold in the model card |

Parameter budget (heads only, frozen backbone): SkyRay 0.06 M, FiLM 0.07 M, attention pool + global MLPs 0.5 M,
dense decoder 0.6 M, CBH head 0.3 M; total ≈ 1.5 M trainable in Stage H's frozen stages.

## 3. CBH bins (ADR-005)

24 bins whose edges contain the ontology's étage thresholds (2,000 m and 6,000 m), log-spaced inside each étage so
the layer-consistency term (ADR-006) is a plain sum over bins:

| Étage | Bins | Edges (m) |
|---|---|---|
| low | 10 | 0, 100, 139, 195, 271, 379, 528, 737, 1028, 1434, 2000 |
| mid | 8 | 2000, 2294, 2632, 3020, 3464, 3974, 4559, 5230, 6000 |
| high | 6 | 6000, 6735, 7560, 8485, 9524, 10691, 12000 |

Bin 24 = "no cloud overhead". Bases above 12,000 m (rare in the CHM15k record) fall into the last height bin. The
predicted height is the median of the distribution over bins (bin centres in log space); the interval is the
10–90 % range. `stratia/model/cbh_bins.py` is the single source of these edges; the model card copies them.

## 4. Contract mapping

| Contract output | Module | Loss (ADR-006) | Supervision |
|---|---|---|---|
| `genus_logits` | `GlobalHeads.genus` | set-valued BCE (partial negatives); soft BCE on Montenegro shares | CCSN, MGCD, SWIMCAT (clear/cloud), Montenegro |
| `etage_logits` | `GlobalHeads.etage` | BCE on étage sets; soft BCE on Montenegro; hierarchy consistency with genus | all genus-labelled sets, Montenegro, Almería (from layers), ceilometer étage |
| `oktas_probs` | `GlobalHeads.oktas` (CORAL) | CORAL with soft cumulative targets (annotator distribution) | Montenegro N; derived cover from sky parsing (P074) |
| `sky_parse_logits` | `DenseDecoder` + `SkyParseHead` | CE, ignore 255 | SWIMSEG family, Almería; glare class from the augmentation's glare mask (P074) |
| `layer_logits` | `DenseDecoder` + `LayerHead` | CE on cloud pixels | Almería layer masks; ceilometer weak labels (P075) |
| `cbh_probs` | `CbhHead` | ordinal CE (soft log-height targets) + CRPS + layer consistency | Eye2Sky pairs (P047); Montenegro h bands as ranges |
| `ood_score` | `OodScore` | none (fitted statistics) | in-domain train features |

Missing labels are masked per sample and per head (P069); every loss returns 0 with a zero gradient when its mask
is empty, so any batch composition is valid.

## 5. Module map

| File | Contents | Phase |
|---|---|---|
| `stratia/model/skyray.py` | `SkyRayEncoder`, `MetaFiLM`, `GeometryEncoder` (fusion with α, FiLM on cls and tokens) | P050, P062 |
| `stratia/model/heads.py` | `AttentionPool`, `CoralHead`, `GlobalHeads` | P063 |
| `stratia/model/losses.py` | set-valued, soft BCE, CORAL, KL, hierarchy consistency, ordinal CE + CRPS | P064, P067 |
| `stratia/model/dense.py` | `DenseDecoder`, `SkyParseHead`, `LayerHead` | P065, P066 |
| `stratia/model/cbh_bins.py`, `stratia/model/cbh.py` | bin edges, `CbhHead`, distribution → median / intervals | P067 |
| `stratia/model/ood.py` | energy + Mahalanobis, glare / night flags | P068 |
| `stratia/model/stratia.py` | `STRATIA` module assembling everything; `FrozenTokenBackbone` for cached tokens | P071 |
| `stratia/data/sampler.py` | task-batch sampler, temperature balancing, label masks | P069 |
| `stratia/model/balance.py` | uncertainty weighting, gradient-conflict monitor | P070 |
| `stratia/train/engine.py` | AMP, accumulation, clipping, EMA, checkpoint/resume, NaN guard, early stopping | P072 |

## 6. Frozen-feature training path (Stages G–H)

Heads train on the P051 caches: `cls` for the global heads, the 32 × 32 patch grid for the dense heads (dense
subset, 8,870 images) and an 8 × 8 pooled grid for the CBH head (P067; 20,942 frames, 1 GB). The ray map is
computed at the matching grid (patch 16 or 64) from the camera, the fitted pose and the Sun; geometry dropout
(P050) applies at batch time. The full model with the backbone in the graph (P077) uses the same heads unchanged:
`STRATIA.forward(image, ray_map, meta)` and `STRATIA.forward_tokens(tokens, cls, ray_map, meta)` share the code
after the backbone.
