# P063 — Global heads

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The three global contract outputs from the geometry-fused tokens: `genus_logits` [N, 12] and `etage_logits` [N, 3]
(multi-label), `oktas_probs` [N, 9] (ordinal), through an attention pool, the CLS token and one shared trunk.

## Inputs / dependencies
P062 `GeometryEncoder` outputs; ADR-005 §1; `stratia/contract.py` class orders and `output_shapes`.

## Work log
1. `stratia/model/heads.py`:
   - `AttentionPool(d_model, n_heads=4)`: LayerNorm on the tokens, one learned query, `nn.MultiheadAttention`
     (batch-first), returns [N, d]; `weights()` exposes the head-averaged attention map for the P093 figures;
     a key-padding mask is accepted for padded token sets.
   - `CoralHead(d_in, n_classes=9)`: one linear projection shared by the 8 thresholds, 8 biases sorted in
     decreasing order at forward time (rank consistency holds for any parameter values), cumulative logits
     P(y > k) → class probabilities by differences, clamped at 0 and renormalised to sum to 1.
   - `GlobalHeads(d_model, hidden=512, dropout=0.1)`: pool ⊕ cls → LayerNorm → Linear → GELU → Dropout → the
     three outputs; returns the cumulative oktas logits for the CORAL loss and the trunk features for the OOD
     score (P068).
2. No BatchNorm (P010); 0.55 M parameters at d = 384, hidden = 512.

## Verification
- `tests/test_heads.py` (4): shapes equal `contract.output_shapes`; oktas probabilities non-negative and summing
  to 1; CORAL cumulative probabilities non-increasing for scrambled bias values and a positive projection moves
  mass to higher classes; attention weights sum to 1 and a padding mask reproduces the unpadded pool; one training
  step gives finite gradients on every parameter.

## Exit criteria
- [x] Shape / contract tests.

## Fit & data-risk notes
- The trunk is shared by three heads with very different label volumes (genus on 11k images, oktas on Montenegro's
  2.5k); the dropout and the Kendall weighting (P070) are the first defences against the big head dominating.
- Sorting the CORAL biases at forward time is branch-free and exportable; the loss acts on the cumulative logits,
  so the sort never fights the optimiser.

## Deviations from plan & why
- None.

## Next phase
P064 — Taxonomy and soft-label losses.
