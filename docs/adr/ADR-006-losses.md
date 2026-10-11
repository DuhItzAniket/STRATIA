# ADR-006 — Losses: set-valued labels, soft labels, ordinal targets, hierarchy and layer consistency

Status: Accepted     Date: 2026-10-11     Phase: P061

## Context
STRATIA's labels are heterogeneous by design (P034): genus *sets* with an `alternatives` flag (a merged class such
as MGCD "altocumulus" = Ac or Cc), single labels from datasets that do not claim to be exhaustive (a CCSN picture
labelled Ac may also contain Cu), expert *distributions* (Montenegro: shares of nine raters per genus, étage, oktas
and height band), pixel masks with an ignore value, and a ceilometer height that is a point measurement of a
quantity the picture shows only as a distribution. A single cross-entropy per head would force every one of these
into a wrong shape. The plan asks for a set-valued loss, KL to annotator distributions, hierarchy consistency,
ordinal CE + CRPS for CBH and a layer-consistency term (PLAN P064, P067).

## Decision
All losses live in `stratia/model/losses.py`, take a per-sample mask and return 0 with a zero gradient when the
mask is empty. Notation: p = sigmoid(logit) for multi-label heads, q = softmax for distributions.

1. **Genus, set-valued (`set_valued_bce`).** For a sample with genus set S and flag `alternatives`:
   - co-present set (alternatives = False): BCE(p_c, 1) for every c ∈ S;
   - alternatives (True): at least one of S is present: −log(1 − Π_{c∈S}(1 − p_c));
   - negatives: BCE(p_c, 0) for c ∉ S, weighted by the dataset's `negative_weight` (1.0 for exhaustive labelling
     such as Montenegro's multi-label votes and the `clear` class, 0.3 for single-label datasets whose label names
     one genus without excluding the others). `clear` is a positive with all genera negative at weight 1.
   The gradient of the alternatives term pushes mass into the set without choosing a member; a sample whose set
   already holds probability 1 contributes no gradient (tested).
2. **Soft labels (`soft_bce`, `kl_to_distribution`).** Montenegro shares are Bernoulli targets per genus and per
   étage: BCE(p, share). Oktas: the annotator distribution over 0–8 is the target of the CORAL head through its
   cumulative form (3); a KL(target ‖ q) term on the class distribution is reported as the agreement metric, not
   optimised separately.
3. **Ordinal (`coral_loss`).** For K ordered classes the head gives K−1 cumulative logits ℓ_k (shared weights,
   ordered biases); the loss is Σ_k BCE(sigmoid(ℓ_k), P(y > k)). Hard labels give 0/1 cumulative targets, a
   distribution gives P(y > k) = Σ_{j>k} t_j. Rank consistency follows from the shared weights (CORAL).
4. **Hierarchy consistency (`hierarchy_consistency`).** From the genus probabilities, the implied étage presence
   is a noisy-OR over the genera whose primary étage is e (ontology): p̂_e = 1 − Π_{g∈e}(1 − p_g). Loss =
   mean_e (p_e − p̂_e)² with p_e = sigmoid(etage_logit), gradients to both heads. `extends` étages of a genus are
   not forced (As, Ns, Cu, Cb may reach other étages; the étage head decides from the picture).
5. **Dense (`dense_ce`).** Cross-entropy with ignore 255, optional class weights; the layer head's targets are
   255 off cloud pixels, so it is trained on cloud pixels only.
6. **CBH (`cbh_loss`).** The target for a ceilometer base h is a distribution over the 24 height bins, Gaussian in
   log height with σ = 0.15 (about one bin), renormalised; "no cloud" is a one-hot on bin 24. Loss =
   cross-entropy(q, target) + λ_crps · CRPS, with CRPS the mean squared difference of the cumulative distributions
   over the 25 ordered classes (no-cloud last; CRPS is computed on the height bins only for cloudy targets and on
   the no-cloud indicator otherwise). λ_crps = 1. The soft target keeps the ordinal structure in the CE term; the
   CRPS term penalises mass far from the truth in proportion to distance, which plain CE does not.
7. **Layer consistency (`cbh_etage_consistency`).** The CBH distribution implies étage-of-lowest-base probabilities
   by summing bins inside each étage (the bin edges contain the thresholds, ADR-005); loss = KL between that
   implied distribution and the étage head's lowest-étage distribution (derived from the three sigmoid outputs:
   P(lowest = low) = p_low, P(lowest = mid) = (1 − p_low)·p_mid, P(lowest = high) = (1 − p_low)(1 − p_mid)·p_high,
   P(none) = the rest, renormalised), gradient to the CBH head only (the étage head has more and cleaner
   supervision). Weight λ_layer = 0.1.
8. **Multi-task combination (P070).** Kendall uncertainty weighting: L = Σ_t exp(−s_t)·L_t + s_t with one learned
   s_t per task; a task-gradient cosine monitor logs conflicts but does not change the weights in v1.

## Consequences
- Partial negatives (weight 0.3) are a hyperparameter of the genus loss; P080 ablates it. Set to 1.0 the loss
  reduces to multi-label BCE on single labels, which is the baseline behaviour.
- The CBH target σ and λ_crps are config fields; the mode-collapse guard of P067 (prediction histogram vs label
  histogram) is the check that they are not too soft.
- Every loss has a toy test with a known gradient direction (P064): mass moves into the set, a satisfied target
  gives zero gradient, CORAL stays rank-consistent, CRPS decreases when mass moves toward the truth.
