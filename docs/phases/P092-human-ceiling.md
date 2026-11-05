# P092 — Human-ceiling comparison

Status: DONE     Date: 2026-11-05     Commit: (this commit)

## Objective
Plan (Stage J, P092): STRATIA as an extra Montenegro annotator; Krippendorff's alpha with and without the model;
a figure. Research design §4: "agreement of the model with a random expert vs expert–expert agreement; alpha with
the model added".

## Inputs / dependencies
- Montenegro annotations (`DATA/Montenegro/annotations.csv`, five raters) through `stratia.data.montenegro` and the
  P035 derivation (`stratia.labels.agreement.derive`: étage set from the altitude class, cover in oktas from N).
- The Gate G4 global heads `20261011T125754Z_global_indomain_e2s_s0_27d0d07c22` (EMA) on the cached features of the
  511 Montenegro validation images (the P090 geometry cache `pool4_locked_indomain_val_*` is reused).
- Validation units only; the locked test was read once at P090 and is not touched.

## Work log
1. `scripts/human_ceiling_model.py`: the model answers each validation image in the raters' coding (étage set =
   the étages with sigmoid > 0.5, joined as the raters' `low|mid`; oktas = the CORAL argmax), is appended to the
   long annotation table as rater `STRATIA`, and the P035 statistics are recomputed: alpha over the humans and over
   humans + model (nominal for the étage set, interval for oktas), the leave-one-out agreement of every rater with
   the majority (étage set) or the median within ±1 okta (cover) of the others, and pairwise agreement model–human
   against human–human. Run `20261105T…_human_ceiling_model` (registry, phase P092).
2. Outputs: `data/human_ceiling_model.parquet`, `docs/data/human_ceiling_model.md`,
   `docs/data/figures/human_ceiling_model.png`.

## Verification
511 validation images, 5 human raters, 2,269 human annotations.

| Quantity | Alpha, humans | Alpha, humans + STRATIA | Model vs the others' majority | Humans (mean; min–max) | Pairwise model–human | Pairwise human–human |
|---|---|---|---|---|---|---|
| étage set | 0.659 | **0.684** | 0.932 (486 units) | 0.893 (0.827–0.955) | 0.829 | 0.769 |
| cover, oktas ±1 | 0.960 | 0.956 | 0.892 (509 units) | 0.909 (0.808–0.964) | 0.857 | 0.874 |

On the étage set the model ranks second of six raters (behind one human, ahead of four) and adding it raises the
panel's alpha; on cover it ranks fifth of six, within the human range, and the panel's alpha is unchanged to the
second decimal. Figure: `docs/data/figures/human_ceiling_model.md` (bars per rater, the model in brown).

## Exit criteria
- [x] STRATIA as an extra Montenegro annotator; alpha with and without the model.
- [x] Figure.

## Fit & data-risk notes
- Validation images only: the model's heads were selected on these rows (early stopping on the mean étage F1 across
  sources), so the human-vs-model comparison is slightly optimistic for the model; the locked-test C2 ratios (0.926 /
  0.963 of the ceiling on test, P090) are the unbiased counterpart and tell the same story.
- The étage set is the raters' coarsest variable (alpha 0.66 among humans); cover is their most consistent one (0.96),
  which is where the model is still below the human mean.
- Leave-one-out units differ per quantity because obscured frames carry no okta.

## Deviations from plan & why
None.

## Next phase
P093 — Error analysis and shortcut re-check.
