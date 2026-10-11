# ADR-005 — Heads: attention pooling, CORAL for ordinal outputs, a shared light dense decoder, binned CBH

Status: Accepted     Date: 2026-10-11     Phase: P061

## Context
The contract (ADR-001) fixes seven outputs of one model: three global (genus, étage, oktas), two dense (sky
parsing, layer) and two global-with-geometry (cloud-base height, reliability). The backbone is a frozen DINOv3
ViT-S/16 for Stages G–H (ADR-002), so the heads must (a) train from cached tokens, (b) stay small enough for a
6 GB card when the backbone is later in the graph, (c) avoid BatchNorm (batch 16 at 512 px, P010), and (d) be
exportable to ONNX without data-dependent control flow.

## Options considered
| Question | Options | Chosen |
|---|---|---|
| Global pooling | CLS only; mean of tokens; one-query attention pool | **Attention pool ⊕ CLS**: the pool can look at the sky where the clouds are, CLS keeps DINOv3's own summary; mean pooling scored below CLS on the probes (P054) |
| Oktas | 9-way softmax; regression; CORAL ordinal | **CORAL** (shared weights, ordered biases): rank-consistent, calibrated against annotator distributions through cumulative targets |
| Étage | derived from genus only; own head | **Own head + consistency term** (ADR-006): étage has supervision genus has not (ceilometer, Almería, Montenegro CL/CM/CH) |
| Dense decoder | per-token linear probe (P057); full DPT with four reassembled levels; light DPT-style trunk | **Light DPT-style trunk** on the last-layer tokens (two fusion blocks, 32² → 128², GroupNorm): a pixel-level output the probe cannot give, multi-level fusion optional once the backbone is in the graph |
| Layer head | separate decoder; shared trunk | **Shared trunk, separate 1 × 1 head**: layers are a cloud-pixel property of the same features |
| CBH output | regression; K-way softmax over log-spaced bins; ordinal distribution | **Distribution over 24 bins + no-cloud class**, bins log-spaced inside each étage so the étage thresholds are bin edges |
| CBH pooling | CLS; zenith ROI crop; zenith-weighted attention over tokens | **Attention with a zenith prior** from the ray map's z channel (uniform prior when geometry is null), ⊕ CLS |
| OOD | max softmax; energy; Mahalanobis; both | **Energy + Mahalanobis** (plan), combined as a z-scored sum; glare and night flags from image statistics |

## Decision
1. `GlobalHeads`: `AttentionPool` (4 heads, one learned query, LayerNorm on tokens) concatenated with the FiLM'd
   CLS → a 2-layer MLP (GELU, dropout 0.1) → `genus_logits` [12], `etage_logits` [3], and a `CoralHead` for oktas
   (one linear projection, 8 ordered biases; probabilities = differences of cumulative probabilities, clamped and
   renormalised).
2. `DenseDecoder`: tokens → [384, 32, 32] → 1 × 1 to 128 → fusion block (3 × 3 conv, GroupNorm, GELU, ×2) →
   bilinear ×2 → 96 channels at 64² → fusion block → ×2 → 64 channels at 128²; `SkyParseHead` and `LayerHead` are
   1 × 1 convolutions on that feature map. ≈ 0.6 M parameters.
3. `CbhHead`: attention logits = linear(token) + prior, prior = −(θ/σ)² with θ the zenith angle from the ray map
   (σ = 20° initial, learnable) and 0 where the ray map is invalid or null; softmax over tokens → pooled ⊕ CLS →
   MLP → 25 logits → softmax. The predicted height is the distribution's median over log-spaced bin centres.
4. CBH bins (docs/architecture.md §3): edges 0, 100, …, 2000 (10 low bins), …, 6000 (8 mid bins), …, 12000 (6 high
   bins); `stratia/model/cbh_bins.py` is the single source and the model card copies it.
5. `OodScore`: energy = −logsumexp(genus_logits); Mahalanobis distance of the pooled feature to the training
   Gaussian (shared covariance, shrinkage 0.1); `ood_score` = mean of the two z-scores (statistics fitted on
   training data, stored in the model card). Night flag: mean luminance below a threshold; glare flag: saturated
   fraction inside 15° of the Sun when geometry is known.
6. No BatchNorm anywhere; GroupNorm(8) in the decoder, LayerNorm elsewhere. Every head is a plain function of its
   inputs (torch.where for the null states), so the ONNX export (P096) traces without branches.

## Consequences
- The dense heads work from the last-layer tokens only in Stages G–H; a multi-level DPT fusion is a P077 option
  (`levels` argument), not a contract change.
- The CBH head needs a token grid for the Eye2Sky frames; Stage G caches an 8 × 8 pooled grid (P067) rather than
  the full 32 × 32 (16 GB).
- The oktas and CBH heads are rank-consistent by construction, which the calibration phase (P089) relies on.
