# P091 — Ablations

Status: DONE     Date: 2026-11-05     Commit: (this commit)

## Objective
Plan (Stage J, P091): ablate the SkyRay geometry (with / without / always), the set-valued and soft targets against
hard labels, the hierarchy loss, the task weighting, the Eye2Sky étage rows, and gather the Stage H variants
(regularisation, joint training, LoRA, CBH geometry, dense heads) into one table. Validation data only: the locked
test sets were read once at P090 and are spent.

## Inputs / dependencies
- `scripts/train_global_heads.py` with three new switches for the five P091 runs: `--p-drop` (geometry dropout
  probability; 1.0 = geometry never given, 0.0 = always given), `--hard-targets` (`harden_targets`: the first listed
  genus of a set or the argmax of a share vector, étages above 0.5 or the single argmax, the argmax okta,
  alternatives dropped), `--kendall` (uncertainty task weighting). `--lambda-h 0` already existed.
- The Gate G4 reference `indomain_e2s` seed 0 (`20261011T125754Z`; its three seeds spread by less than 0.01 in mean
  étage F1), the P080 regularisation runs, the P073 LODO runs with and without the Eye2Sky étage rows, P076 joint
  training, P077 LoRA, P075 CBH variants, P074 dense variants, all from `data/stage1_global.parquet`,
  `data/stage4_joint.parquet`, `data/stage5_lora.parquet`, `data/cbh_head.parquet` and `data/dense_heads.parquet`.
- `scripts/ablations.py` -> `data/ablations.parquet`, `docs/data/ablations.md`; `scripts/reproduce_tables.py` ->
  `paper/tables/tab5_ablations.tex` and the `\nAbl*` macros of `paper/numbers.tex`.

## Work log
1. Five runs on the cached features (P091, `--eye2sky-etage`, seed 0, 4–6 min each on the RTX 4050):
   `abl_nogeo` (p_drop 1.0), `abl_geoalways` (p_drop 0.0), `abl_hard`, `abl_noh` (lambda_h 0), `abl_kendall`.
2. Mean étage macro-F1 over the sources (reference 0.817) and mean genus macro-F1 (0.719):

   | Variant | Étage F1 | Genus F1 |
   |---|---|---|
   | no geometry (p_drop 1.0) | 0.809 (−0.008) | 0.720 (+0.001) |
   | geometry always (p_drop 0.0) | 0.829 (+0.011) | 0.709 (−0.010) |
   | hard targets | 0.794 (−0.023) | 0.729 (+0.010) |
   | no hierarchy loss | 0.813 (−0.005) | 0.712 (−0.007) |
   | Kendall task weighting | 0.814 (−0.003) | 0.716 (−0.002) |
   | without the Eye2Sky étage rows (P073 `indomain`) | 0.823 (+0.005) | 0.738 (+0.019) |

   The SkyRay geometry brackets the reference within about a point either way: on these sources the pixels carry
   the signal, and geometry dropout costs nothing against "always given" while keeping unknown geometry a valid
   input. Collapsing the set-valued and soft targets to one hard label is the largest single loss on étage
   (−0.023) with a small genus gain (+0.010): the sets are what the étage head learns from. The hierarchy loss and
   the task weighting are within the seed spread.
3. Eye2Sky étage rows in the LODO folds (held-out étage F1 with / without): CCSN 0.489 / 0.486, MGCD 0.749 /
   0.871, Montenegro 0.360 / 0.258, Almería 0.593 / 0.327. The ceilometer-derived rows are what make the all-sky
   folds (Montenegro, Almería) transfer and what cost MGCD, whose étages follow its seven sky types; the four-fold
   mean goes 0.486 -> 0.548, which is the LODO number of C1. In-domain they cost 0.005 étage and 0.019 genus.
4. Stage H variants carried into the table: head dropout 0 / 0.3, weight decay 0 / 0.2, no EMA / EMA 0.999 (all
   within ±0.013 étage); joint training 0.812 / 0.799 étage with layer mIoU 0.544 / 0.421 against 0.559 for the
   staged dense head; LoRA 0.810 after / 0.805 before; the CBH head with and without geometry at test time (566 m
   both; 551 m without geometry dropout but 765 m once the geometry is zeroed, the case the deployment needs);
   dense heads with and without layers.
5. A bug in the assembler's first draft gave the "without Eye2Sky rows" LODO rows the "with" numbers
   (`startswith("with")` matched "without"); fixed and covered by the doc numbers above.
6. Tests: `tests/test_stage_j.py` (`harden_targets` on sets, shares, ties and masked rows; the assembler's
   latest-run reader).

## Deviations from the plan
- Backbone size is not ablated (ViT-S/16 was fixed by the laptop and edge budget; the ViT-B/L probes of P054 are
  the comparison the paper makes). Augmentation ablations do not apply to cached-feature training (the heads see one
  transform, `ffr1`). One seed per variant; the reference's three-seed spread (< 0.01) is the noise floor quoted.

## Outputs
- `data/ablations.parquet` (48 rows), `docs/data/ablations.md`, `docs/data/stage1_global.md` (the five runs),
  `paper/tables/tab5_ablations.{tex,md}`, `paper/numbers.tex` (`\nAblNoGeo`, `\nAblGeoAlways`, `\nAblHard`,
  `\nAblNoH`, `\nAblKendall`, `\nAblRef`).
