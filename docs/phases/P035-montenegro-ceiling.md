# P035 — Montenegro soft labels & human ceiling

Status: DONE     Date: 2026-10-08     Commit: (this commit)

## Objective
Translate the Montenegro raters' codes into the labels STRATIA predicts, keep them as per-image distributions, and
measure how well the raters agree with each other on those labels: the ceiling a model is compared with (C2).

## Inputs / dependencies
P016 annotation loader (11,191 annotations, 9 raters, 2,522 images); P033 ontology (altitude classes → étages,
C_L / C_M / C_H codes → genera); P026 (rater disagreement first seen there).

## Work log
1. `stratia/labels/agreement.py`: Krippendorff's alpha (nominal, ordinal, interval; missing values allowed), raw
   pairwise agreement, leave-one-rater-out agreement against the majority or median of the others, per-annotation
   derivation (étage set, genus set, oktas, obscured, height band, cloud), per-image soft targets, the ceiling table
   and the report.
2. `scripts/montenegro_ceiling.py`: writes `data/montenegro_targets.parquet` (per image: étage and genus shares,
   oktas and height-band distributions, obscured and cloud shares), `data/montenegro_ceiling.parquet`,
   `docs/data/montenegro_ceiling.md`.
3. Tests (`tests/test_agreement.py`, 3): alpha on a hand-computed example and its invariances (binary interval and
   ordinal equal nominal; perfect = 1; no pairs = NaN; near beats far), pairwise and leave-one-out, derivation and
   soft targets through the ontology, ceiling table and report.

## Verification
- Run: `python scripts/montenegro_ceiling.py` (89 s). Units with at least two raters: 2,270 of 2,522 images.

| Quantity | alpha | Pairwise | Leave-one-rater-out |
|---|---|---|---|
| raw altitude class (5) | 0.34 | 56 % | 71 % |
| raw C_L / C_M / C_H | 0.37 / 0.25 / 0.26 | 50 / 47 / 53 % | 67 / 65 / 67 % |
| total cover N, oktas 0–8 (interval; ±1 okta) | 0.91 | 84 % | **87.3 %** |
| height code h (ordinal; exact / ±1 band) | 0.36 | 40 / 74 % | 37 / 76 % |
| **étage set** (exact) | 0.54 | 75 % | **85.2 %** |
| étage low / mid / high present | 0.62 / 0.20 / 0.39 | 83 / 89 / 87 % | 88 / 93 / 91 % |
| genus set (exact) | 0.28 | 33 % | 58 % |
| genus present, per genus | 0.07 (Cc) … 0.55 (Ci) | 72–98 % | 80–99 % |
| cloud present (N > 0) | 0.67 | 92 % | 95 % |

- **Human ceiling for C2:** a rater agrees with the majority of the others on the étage set in 85.2 % of images and
  on total cover within ±1 okta in 87.3 %; 90 % of these (76.7 % and 78.6 %) is the bar the model must reach on the
  same images against the rater majority.
- Tests: 3 new; `python -m pytest -q`: all pass; `ruff check .` clean.

## Exit criteria
- [x] Ceiling table: `docs/data/montenegro_ceiling.md` (alpha, pairwise, leave-one-out for every raw and derived
  quantity; per-rater rows).
- [x] Soft labels: `data/montenegro_targets.parquet`.

## Fit & data-risk notes
- **Total cover is the one quantity raters agree on well** (alpha 0.91); étage is moderate (0.54); genus from the
  codes is poor (0.28, with per-genus alphas as low as 0.07 for Cc and 0.09 for Cb): Montenegro can test cover and
  étage, and only weakly genus. The paper should quote C2 on étage and cover, as pre-registered, and treat genus on
  Montenegro as a soft, exploratory comparison.
- **Height codes have no consensus** (alpha 0.36, exact leave-one-out 37 %), confirming P026: image-only cloud-base
  "labels" are distributions; the ceilometer remains the CBH reference.
- The "exact set" agreement for étage (85 %) is above the raw altitude-class agreement (71 %) because the ontology
  folds "clouds of vertical development" into the low étage, where raters disagreed only on the name.
- Soft targets are per-image distributions; the loss uses them as such (P030: never resampled).

## Deviations from plan & why
- The plan names "leave-one-annotator-out agreement"; it is computed against the majority (or median) of the other
  raters per image rather than by retraining anything, which is what a model-vs-rater comparison will also do.

## Next phase
P036: Ceilometer → targets.
