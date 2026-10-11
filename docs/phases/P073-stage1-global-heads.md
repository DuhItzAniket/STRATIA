# P073 — Stage-1: global heads on cached features

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Train the global heads (genus, étage, oktas) on the frozen DINOv3 features of every labelled recognition image,
with the Stage G losses, sampler and engine, and report the fit per dataset (train / val curves) and the numbers
the gate reads: in-domain étage and genus macro-F1, the Montenegro agreement against the human ceiling, and the
leave-one-dataset-out étage macro-F1 against the P054 probe.

## Inputs / dependencies
P051 caches (+ an 8 × 8 pooled token grid of every manifest image and CBH frame, built here: 3.6 GB, 7.6 min);
P034 labels, P035 Montenegro shares, P038 / P039 v2 splits; P047 pairs (the ceilometer's lowest-base étage as a
partial label for the Eye2Sky frames); P062–P064 modules, P069 sampler, P070 weighting, P072 engine.

## Work log
1. `stratia/train/data.py`: `global_rows` (one row per image with genus set / alternatives / per-class known-negative
   weights, étage target with per-column weights, oktas distribution, masks; in-domain or LODO split; optional
   Eye2Sky rows whose lowest-base étage is a positive, the étages below it weak negatives (0.3) and the ones above
   unknown), `GeometryCache` (ray maps at the pooled grid for the calibrated rows, cached on disk), `GlobalDataset`,
   `collate`, native-class scoring (`native_class_sets`, `native_scores`: noisy-OR of a native class's genera).
2. `stratia/train/global_eval.py`: `GlobalModel` (the STRATIA model without dense / CBH heads), `losses_for`
   (set-valued genus BCE with per-class weights, soft étage BCE with column weights, CORAL oktas, hierarchy term),
   `predict`, `score_dataset` (étage macro-F1 on rows with exhaustive étage labels, plus the val-tuned-threshold
   variant and the unsupported false-positive rate; genus macro-F1 and accuracy over native classes; Montenegro
   oktas MAE / within-1 / KL and the agreement with the majority étage set and the majority cover ±1 okta as a
   ratio to the P035 leave-one-out ceiling; lowest-étage accuracy on the Eye2Sky partial rows).
3. `scripts/train_global_heads.py`: in-domain or `--lodo <fold>`, `--eye2sky-etage`, geometry dropout 0.3, square-root
   dataset balance, AdamW + cosine, EMA 0.99, early stopping on the val mean étage macro-F1 (patience 6), per-epoch
   scores on val and on a fixed 1,000-row-per-dataset train subset (the fit report), `data/stage1_global.parquet`,
   `docs/data/stage1_global.md`, curves per run.
4. Two scoring rules fixed during the stage: a row counts as exhaustively étage-labelled only when every column
   weight is 1 (Eye2Sky partial rows had leaked into the mean through their "high" labels), and a dataset without any
   positive étage (SWIMCAT clear patches) is not scored on étage. The Eye2Sky-étage runs were rerun under the final
   rules; the numbers below are from those runs.

## Verification
In-domain (grouped val), `indomain_e2s`, best epoch 11 of 20 (seeds 0 / 1 / 2: 0.817 / 0.817 / 0.804 mean étage F1):

| Dataset | Étage macro-F1 (train → val) | Genus macro-F1 (train → val) | Other |
|---|---|---|---|
| CCSN | 0.96 → **0.802** | 0.88 → **0.626** (probe best 0.646, ViT-L) | 11 genera + contrail |
| MGCD | 1.00 → 1.000 | 1.00 → **0.812** (probe best 0.831) | low / high / clear only |
| Montenegro | 0.78 → **0.692** | shares | étage-set agreement 0.892 (ceiling 0.852 → **ratio 1.047**); oktas ±1 0.896 (ceiling 0.873 → **1.027**); oktas MAE 0.49 |
| Almería | 0.92 → **0.775** | — | étage from masks |
| Eye2Sky frames | — | — | lowest-étage accuracy 0.788 (partial labels) |
| mean | 0.92 → **0.817** | 0.94 → 0.719 | |

Leave-one-dataset-out étage macro-F1 on the held-out source's val units (fixed 0.5 thresholds; in brackets the
per-column thresholds chosen on the same val rows, the optimistic rule the probes' C selection corresponds to):

| Held-out | Sources only | + Eye2Sky étage | P054 best probe (ViT-B @ 224) |
|---|---|---|---|
| CCSN | 0.486 | 0.489 [0.565] | 0.391 |
| MGCD | 0.871 | 0.749 [0.946] | 0.995 |
| Montenegro | 0.258 | 0.360 [0.462] | 0.516 |
| Almería | 0.327 | 0.593 [0.630] | 0.429 |
| **mean** | 0.486 | **0.548** [0.651] | **0.583** |

- Reading: in-domain the head sits 0.02 below the best frozen probes on CCSN and on MGCD (the probes'
  C is chosen on val, the head's epoch too); the Montenegro agreement is above the leave-one-out human ceiling on
  both quantities. **Leave-one-dataset-out is the weak result:** the mean is 0.035 below the P054 reference at the
  pre-registered rule. The diagnostic (per-column rates on the held-out val) shows why: the étage *calibration*
  does not transfer across sources (on Montenegro the head predicts "mid" for 42 % of the images against 2 % in
  the majority labels and "low" for 17 % against 61 %; on Almería it never predicts "mid" against 46 % present),
  while the *ranking* partly does (AUROC 0.78 / 0.88 for low / high on Montenegro). "Mid" means different things
  across sources: a genus (Ac / As / Ns) in CCSN, a per-pixel layer in Almería's masks, a rater vote in Montenegro,
  absent from MGCD's étage labels. The Eye2Sky partial labels help Almería and Montenegro (+0.27, +0.10) and cost
  MGCD (−0.12: the ceilometer's "high" rows push mid / high positives into a fold whose val has none).
- Fit: train étage F1 0.92 against val 0.82 (gap 0.10, "ok" by the PLAN §4 rule, overfit-leaning on CCSN at 0.16);
  the LODO folds overfit the training sources within 1–6 epochs (Montenegro and Almería peak at epoch 1), which
  P080's LODO variants address.

## Exit criteria
- [x] Fit report: train / val curves per dataset (`curves.json` per run, summarised above and in P081).

## Fit & data-risk notes
- The Montenegro agreement is measured against the *majority* set and cover, not against a random rater; the
  ceiling (0.852 / 0.873) is each rater against the others' majority, so the two are comparable in kind, and the
  model exceeds it. The raw per-rater sets exist (P035) and P089 / P093 can compute the rater-vs-model Krippendorff
  α for the paper.
- The LODO étage task mixes label definitions; the paper must say so when it reports C1, and P093's error analysis
  should split the LODO error by "mid".

## Deviations from plan & why
- The ceilometer étage of the Eye2Sky frames was added as a training label (the plan lists only the recognition
  sources); it is reported as a separate column so its effect is visible.

## Next phase
P074 — Stage-2: dense heads.
