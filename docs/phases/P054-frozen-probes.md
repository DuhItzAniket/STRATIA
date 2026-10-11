# P054 — B3: frozen probes

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Linear and k-nearest-neighbour probes on the frozen P051 features of DINOv2 S/B and DINOv3 S/B/L: the recognition
baselines the STRATIA head has to beat in-domain (criterion C1's reference) and under leave-one-dataset-out.

## Inputs / dependencies
P051 caches; P038 grouped in-domain splits; P039 LODO folds (amended here, see below); P034 labels (native classes,
étage sets); P040 test lock.

## Work log
1. `stratia/baselines/tasks.py`: the label tables per protocol. In-domain recognition = the native single-label
   classes of CCSN (11), MGCD (7), SWIMCAT (5) and Montenegro (5 altitude classes), grouped train → grouped val.
   LODO étage = the WMO étage sets (multi-label low / mid / high, empty = clear) for the CCSN, MGCD, Montenegro and
   Almería folds, train on the other sources' rows, score on the held-out source. Test rows are never returned
   unless asked for explicitly (P090).
2. `stratia/baselines/probes.py`: linear probe = L2-regularised logistic regression on standardised features, `C`
   from {0.01, 0.1, 1, 10} chosen on val (one binary probe per étage column, 0.5 threshold); k-NN = cosine, k = 20,
   distance-weighted votes.
3. **LODO amendment.** The P039 folds had only train and test roles, so a LODO number could not be reported without
   opening the locked test sets. `configs/splits.yaml` is now version 2: each fold's held-out source keeps its
   in-domain val units as the fold's `val` role and the rest stays `test`. The five LODO test entries of
   `test_lock.json` were re-locked (`scripts/lock_test_sets.py --force`); the ten in-domain entries are unchanged.
   Research design §8 records the amendment.
4. **Metric fix.** The MGCD val units carry only three étage patterns (cumulus → low, cirrus → high, clear → none;
   altocumulus, stratocumulus, cumulonimbus and mixed have no étage set under the P034 "nothing invented" rule),
   so the mid column has no positives. The first macro-F1 skipped that column when nothing was predicted and scored
   it 0 after a single false positive: DINOv3 ViT-B read 0.995 and ViT-S 0.649 for the same low/high separation.
   `multilabel_macro_f1` now averages the columns with support in the truth, and the false-positive rate on
   unsupported columns is a separate number (`*_unsupported_fp` in the parquet; 0.00–0.24 for the linear probes).
5. `scripts/baseline_probes.py`: one run per (backbone, size) in the registry (phase P054), rows in
   `data/baseline_probes.parquet`, tables in `docs/data/baseline_probes.md`.

## Verification
- `tests/test_baselines.py` (5): étage matrix and both macro-F1 conventions, linear and k-NN probes on separable
  blobs (single- and multi-label, a constant column handled), the recognition and LODO tables from synthetic tables.
- Linear probe, macro-F1 on val (full tables in `docs/data/baseline_probes.md`):

| Backbone @ size | CCSN | MGCD | SWIMCAT | Montenegro | In-domain mean | LODO étage mean |
|---|---|---|---|---|---|---|
| DINOv3 ViT-B/16 @ 224 | 0.584 | 0.825 | 1.000 | 0.651 | **0.765** | **0.583** |
| DINOv3 ViT-L/16 @ 224 | 0.646 | 0.786 | 1.000 | 0.621 | 0.763 | 0.519 |
| DINOv3 ViT-S/16 @ 512 | 0.586 | 0.831 | 1.000 | 0.600 | 0.754 | 0.435 |
| DINOv3 ViT-S/16 @ 224 | 0.543 | 0.778 | 1.000 | 0.581 | 0.726 | 0.522 |
| DINOv2 ViT-B/14 @ 224 | 0.580 | 0.765 | 1.000 | 0.607 | 0.738 | 0.563 |
| DINOv2 ViT-S/14 @ 224 | 0.570 | 0.810 | 1.000 | 0.573 | 0.738 | 0.400 |

  Val sizes: CCSN 255, MGCD 959, SWIMCAT 79, Montenegro 511; LODO val: CCSN 236, MGCD 438, Montenegro 511,
  Almería 84. The linear probe beats k-NN on every cell. SWIMCAT is saturated (79 patches, 5 classes).
- Reading: DINOv3 ViT-B is the best frozen recognition baseline on the mean; ViT-L is best on CCSN (0.646). LODO étage
  is far below in-domain everywhere except the MGCD fold (three well-separated patterns): CCSN 0.39–0.50,
  Montenegro 0.27–0.52, Almería 0.23–0.44, which is the cross-dataset gap criterion C1 targets.

## Exit criteria
- [x] Table (`docs/data/baseline_probes.md`, 48 rows in the parquet for this phase).

## Fit & data-risk notes
- `C` is chosen on the same val split the number is reported on (the test sets stay locked); the choice is one of
  four values, and the same rule applies to every backbone, so the ranking is fair but the absolute numbers are
  slightly optimistic. The final comparison (P090) runs once on test.
- The Almería LODO val holds 84 rows from a handful of days: its numbers carry a wide interval and are reported, not
  interpreted on their own.

## Deviations from plan & why
- The LODO split now has a val role (amendment above); the plan's folds had none, which would have forced test use.
- Macro-F1 over supported columns (work log 4); the plain convention is kept for single-label tasks.

## Next phase
P055 — B4: CLIP / SigLIP 2.
