# P068 — OOD / quality score

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The contract's `ood_score`: a reliability number a host can threshold, built from the energy of the genus logits
and the Mahalanobis distance of the pooled feature to the training distribution, plus glare and night flags; its
AUROC for held-out-domain against in-domain inputs on cached features.

## Inputs / dependencies
P051 cache (`dinov3_vits16` @ 512 cls); P054 probe recipe (the genus logits in this phase come from a logistic
probe, the trained global head will provide them in Stage H); P045 ray maps (glare flag); ADR-005 §5.

## Work log
1. `stratia/model/ood.py`: `energy(logits)`; `OodScore(d)` with `fit(features, logits)` (mean, shared covariance
   with shrinkage 0.1 → precision; mean / std of both scores on the training rows, all as buffers), `mahalanobis`,
   `forward` = the mean of the two z-scores, [N, 1]; `auroc` (rank statistic with ties); `night_flag` (mean
   luminance < 0.08); `glare_flag` (saturated share ≥ 0.3 in the ray-map patches within 15° of the Sun; False
   without geometry).
2. `scripts/ood_eval.py`: leave-one-source-out over CCSN, MGCD, Montenegro (probe + score fitted on the other two
   sources' train rows; in = their val rows, out = the held-out source's val rows), plus 2,000 Eye2Sky all-sky
   frames and 2,000 SWIM patches as out sets. `data/ood_eval.parquet`, `docs/data/ood_eval.md`, one registry run.

## Verification
- `tests/test_ood.py` (2): shifted features score higher (AUROC > 0.95 on a Gaussian toy), `auroc` is 1 / 0.5 on
  separated / identical scores, energy falls with confidence, buffers survive a state-dict round trip; the flags on
  a dark frame and on a saturated patch at the Sun (and not without geometry).
- AUROC on cached DINOv3 ViT-S @ 512 features:

| Held-out source | Combined | Energy only | Mahalanobis only | vs Eye2Sky frames | vs SWIM patches |
|---|---|---|---|---|---|
| CCSN | 0.999 | 0.455 | 1.000 | 0.996 | 0.998 |
| MGCD | 0.961 | 0.872 | 0.963 | 0.991 | 0.874 |
| Montenegro | 0.974 | 0.755 | 0.968 | 0.939 | 0.911 |

- Reading: the Mahalanobis term does the work (a held-out camera is a different feature cloud, which P029's
  dataset-identity probe already said); the energy term adds nothing and even inverts on CCSN (consumer photos get
  *confident* genus logits from a probe trained on all-sky imagers). The combined score clears the supporting
  criterion (AUROC ≥ 0.80) in every fold. The flip side for the paper: a score that reads camera identity will call
  any new camera "uncertain", including the owner's B0268, until Stage I's adaptation adds its features to the fit.

## Exit criteria
- [x] AUROC on held-out domain vs in-domain.

## Fit & data-risk notes
- The score's statistics must be refitted whenever the training mix changes (a buffer, saved with the model, copied
  to the model card with the threshold).
- Deferrable phase done in its minimal form: no temperature sweep for the energy, no per-class Gaussians; the
  numbers above say neither is needed for the criterion.

## Deviations from plan & why
- Genus logits from a probe rather than the trained head (Stage H re-runs the script on the head's outputs).

## Next phase
P069 — Multi-dataset sampler.
