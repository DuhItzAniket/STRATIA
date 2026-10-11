# P064 — Taxonomy and soft-label losses

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The losses that let heterogeneous labels train the same heads: genus sets with alternatives and partial negatives,
expert shares, ordinal targets (hard and soft), hierarchy consistency between genus and étage, dense cross-entropy
with ignore; plus the CBH losses specified in ADR-006 (used by P067) and the bin module they rest on.

## Inputs / dependencies
ADR-006; P034 labels (`genus_set`, `genus_alternatives`, `etage_set`), P035 Montenegro shares, configs/ontology.yaml
(genus → étage, étage heights); P063 CORAL cumulative logits.

## Work log
1. `stratia/model/losses.py`: `set_valued_bce` (co-present sets: BCE per member; alternatives:
   −log(1 − Π(1 − p_c)); complement negatives weighted by a per-sample `negative_weight`), `soft_bce`,
   `hierarchy_consistency` (noisy-OR genus → étage, squared error, gradient to both heads), `cumulative_targets`
   + `coral_loss` (hard labels or distributions), `kl_to_distribution`, `dense_ce` (ignore 255, upsamples logits
   to the target size), `cbh_loss` (CE on a log-height Gaussian target + CRPS over the ordered classes),
   `cbh_etage_consistency` (KL from the CBH-implied lowest étage to the étage head's, étage detached), with the
   helpers `lowest_etage_from_etage_logits` and `cbh_implied_etage`. Every loss is a masked mean that returns an
   exact 0 with a zero gradient for an empty mask.
2. `stratia/model/cbh_bins.py`: the 24 étage-aligned log bins (edges 0, 100 … 2000 … 6000 … 12000),
   `bin_of`, `soft_target`, `median_height_m`, `interval_m`, `ETAGE_OF_BIN`.
3. `genus_etage_matrix()` reads the ontology once: Ci/Cc/Cs → high, Ac/As/Ns → mid, Sc/St/Cu/Cb → low, clear and
   contrail → none (the `extends` étages are not enforced, ADR-006 §4).

## Verification
- `tests/test_losses.py` (6), each a toy case with a known gradient direction:
  - set-valued: an MGCD "Ac or Cc" sample pushes Ac and Cc up by the same amount and Cu down; a set already
    satisfied by one member gives no positive gradient under alternatives and still pushes the other member under
    co-presence; an empty mask gives loss 0 and zero gradients.
  - soft BCE moves toward the share and is flat at 0.5; hierarchy consistency with Cu certain pushes the étage
    head's low logit up and mid / high down toward the implied 0.
  - CORAL cumulative targets for hard labels and distributions; the gradient raises thresholds below the label and
    lowers those above; KL of a distribution to itself is 0.
  - dense CE returns 0 on an all-ignore target and upsamples 4 × 4 logits to an 8 × 8 target.
  - bins: 24 bins, edges contain 2,000 and 6,000 m, 10/8/6 per étage, `bin_of` at the edges, soft targets sum to 1
    and peak at the truth's bin, median and interval bracket the truth.
  - CBH loss prefers mass one bin from the truth over mass in the high étage; CRPS falls as mass moves toward the
    truth at equal CE; the consistency term pulls CBH mass out of the high bins and the no-cloud class when the
    étage head says "low".

## Exit criteria
- [x] Toy-case tests with correct gradients.

## Fit & data-risk notes
- `negative_weight` is the one loss knob with a real modelling claim (single-label datasets are not exhaustive);
  0.3 is the prior, P080 ablates 0.3 vs 1.0.
- The soft CBH target width σ = 0.15 in log height is about one bin; too wide a σ hides mode collapse behind a
  smooth target, which is why P067 checks the prediction histogram against the label histogram.

## Deviations from plan & why
- The plan lists "KL to annotator distributions" as a training loss; for the multi-label heads a per-class BCE on
  the shares is the same objective per Bernoulli, and for oktas the KL is reported as the metric while CORAL's
  cumulative form carries the distribution into training (ADR-006 §2–3).

## Next phase
P065 — Sky-parsing head.
