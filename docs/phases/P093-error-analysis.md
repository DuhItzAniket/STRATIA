# P093 — Error analysis and shortcut re-check

Status: DONE     Date: 2026-11-05     Commit: (this commit)

## Objective
Plan (Stage J, P093): confusions (St/Sc, Ci/Cs, As/Ns), glare cases, a failure gallery, attention maps; re-run the
dataset-identity probe on the final features. Gallery and notes.

## Inputs / dependencies
- The Gate G4 global heads `20261011T125754Z_global_indomain_e2s_s0_27d0d07c22` (EMA) on the validation rows
  (cached features, the P090 geometry cache); the native-class mapping of P034 (`native_class_sets`,
  `native_scores`) for CCSN and MGCD; `data/robustness.parquet` (P088) for the glare corruption; the B0268 zero-shot
  scores (P083) for the glare class; `stratia.data.shortcuts` (P029) for the identity probe; `cache/img768` for the
  gallery.

## Work log
1. `scripts/error_analysis.py` -> `data/error_analysis.parquet`, `docs/data/error_analysis.md`,
   `docs/data/figures/genus_confusion_{ccsn,mgcd}.png`, `docs/data/figures/failure_gallery.jpg`; run registered
   under phase P093.
2. Genus confusions on the native classes: CCSN 11 classes, 255 validation rows, accuracy 0.643; MGCD 6 classes,
   789 rows, accuracy 0.857. The named pairs on CCSN: St/Sc swapped on 18.5 % of their rows (6 St->Sc, 4 Sc->St),
   Ci/Cs 7.0 %, As/Ns 6.4 %; St is the weakest class (recall 0.24: 6 of 21 go to Sc, 5 to Ns), Ci the second
   (0.43), Ct, Cb, Cc, Cs, Cu and Ns are above 0.75.
3. Étage presence (exhaustive étage rows, threshold 0.5): the mid étage is the one the model misses — 35 % of true
   mid rows on CCSN, 41 % on Almería, 46 % on Montenegro (where mid is 2 % of rows) — while it over-predicts low on
   Almería (spurious 35 %). MGCD's étages are a function of its native class and come out exact; Montenegro's
   exact-set agreement is 0.892.
4. Sun glare: under the P088 glare corruption the étage F1 falls 0.838 -> 0.760, the sky mIoU 0.901 -> 0.816 and the
   clear-sky cloud fraction rises 0.29 -> 0.47 (glare is parsed as cloud); on the B0268 adaptation days the glare
   class is never predicted (0.0 % of pixels on the frames the raters called strong glare).
5. Failure gallery: the 16 exhaustive-étage validation frames with the largest étage error. Eleven of the sixteen
   are CCSN; several carry a native label that disagrees with what the picture shows (an altocumulus field labelled
   St, cumulus from an aircraft labelled Cc), which is the P026/P040 label noise rather than a model error; the
   Almería frames are low+high mixtures where the model answers one étage.
6. Dataset identity on the model's pooled features (512-d) against the backbone's CLS features (384-d): linear
   probe over six sources, balanced (79 per source), units held out with every source on both sides: 1.000 for both
   (chance 0.167). The heads do not remove source identity from the representation (P029 found 99.7 % on the
   backbone); the OOD score relies on exactly that (P068).
7. Attention maps were not produced: the heads' attention pool over the 8 × 8 grid is not kept at inference and the
   sky-parse mask is the per-pixel view the paper uses.

## Verification
`docs/data/error_analysis.md` (tables and figures); run id in `runs/registry.csv` (phase P093, name
`error_analysis`). The first probe run held out Montenegro's single validation block whole and both feature sets
scored 0.580 for the same reason; the script now keeps every source on both sides of the split.

## Exit criteria
- [x] Confusions including the named pairs; glare cases; failure gallery; notes.
- [x] Dataset-identity probe re-run on the final features.
- [ ] Attention maps — not produced (see 7).

## Fit & data-risk notes
- St/Sc and the mid étage are where the model and the labels disagree most; both are also where the P026 label
  audit found the most conflicts, so part of this "error" is label noise the paper should state.
- Validation rows: the heads were early-stopped on them; the numbers describe behaviour, not generalisation (P090).

## Deviations from plan & why
Attention maps dropped (not stored by the architecture at inference); the gallery and the masks stand in.

## Next phase
P094 — Efficiency.
