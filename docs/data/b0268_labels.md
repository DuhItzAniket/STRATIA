# B0268 labels and rater agreement (P041)

393 frames of the owner's Arducam B0268 (sky logger, 4656 x 3496, zenith mount, site BLR01) over 2026-10-18 (120), 2026-10-19 (120), 2026-10-20 (120), 2026-10-21 (33), labelled per image by 2 annotators (A, B) with the sheet of `DATA/B0268/labels/` and translated into the labels STRATIA predicts by `stratia/labels/b0268_labels.py`. alpha = Krippendorff's alpha (1 = perfect, 0 = chance); pairwise = share of rater pairs that agree; leave-one-out = a rater against the majority (or median) of the others, the bar a model is compared with (the B0268 counterpart of criterion C2).

## Annotators A and B (the P041 pair)

| Quantity | Level | Units | alpha | Pairwise | Leave-one-out | Note |
|---|---|---|---|---|---|---|
| oktas 0-8 | interval | 393 | 0.808 | 0.771 | 0.771 | pairwise and leave-one-out within ±1 okta |
| oktas 0-8, exact | interval | 393 | 0.808 | 0.430 | 0.430 |  |
| raw low_cloud | nominal | 393 | 0.478 | 0.949 | 0.949 |  |
| raw mid_cloud | nominal | 393 | 1.000 | 1.000 | 1.000 |  |
| raw high_cloud | nominal | 393 | 0.696 | 0.835 | 0.835 |  |
| raw lowest_base_guess | nominal | 393 | 0.349 | 0.949 | 0.949 |  |
| raw obstruction | nominal | 393 | 0.623 | 0.982 | 0.982 |  |
| raw sun_glare | ordinal | 393 | 0.487 | 0.679 | 0.679 | none < mild < strong |
| étage set | nominal | 393 | 0.845 | 0.911 | 0.911 | exact set from the three genus columns |
| étage low present | nominal | 393 | 0.349 | 0.949 | 0.949 |  |
| étage mid present | nominal | 393 | 1.000 | 1.000 | 1.000 |  |
| étage high present | nominal | 393 | 0.909 | 0.957 | 0.957 |  |
| genus set | nominal | 393 | 0.673 | 0.789 | 0.789 | exact set; 'unsure' rows excluded |
| genus Ci present | nominal | 393 | 0.909 | 0.957 | 0.957 |  |
| genus Cc present | nominal | 393 | n/a | 1.000 | 1.000 |  |
| genus Cs present | nominal | 393 | -0.064 | 0.878 | 0.878 |  |
| genus Ac present | nominal | 393 | 1.000 | 1.000 | 1.000 |  |
| genus As present | nominal | 393 | n/a | 1.000 | 1.000 |  |
| genus Ns present | nominal | 393 | n/a | 1.000 | 1.000 |  |
| genus Sc present | nominal | 393 | n/a | 1.000 | 1.000 |  |
| genus St present | nominal | 393 | n/a | 1.000 | 1.000 |  |
| genus Cu present | nominal | 393 | 0.474 | 0.949 | 0.949 |  |
| genus Cb present | nominal | 393 | 1.000 | 1.000 | 1.000 |  |
| cloud present (oktas > 0) | nominal | 393 | n/a | 1.000 | 1.000 |  |
| clear | nominal | 393 | n/a | 1.000 | 1.000 |  |

### Raters against the others (étage set and total cover)

| Rater | Units (étage) | Étage agreement | Units (oktas) | Oktas within ±1 |
|---|---|---|---|---|
| A | 393 | 0.911 | 393 | 0.771 |
| B | 393 | 0.911 | 393 | 0.771 |
| all | 786 | 0.911 | 786 | 0.771 |

## All annotators (A, B, C, D; sensitivity, not the protocol pair)

| Quantity | Level | Units | alpha | Pairwise | Leave-one-out |
|---|---|---|---|---|---|
| oktas 0-8 | interval | 393 | 0.834 | 0.824 | 0.847 |
| oktas 0-8, exact | interval | 393 | 0.834 | 0.460 | 0.454 |
| raw low_cloud | nominal | 393 | 0.491 | 0.955 | 0.964 |
| raw mid_cloud | nominal | 393 | 1.000 | 1.000 | 1.000 |
| raw high_cloud | nominal | 393 | 0.279 | 0.533 | 0.607 |
| raw lowest_base_guess | nominal | 393 | 0.460 | 0.960 | 0.966 |
| raw obstruction | nominal | 393 | 0.501 | 0.976 | 0.978 |
| raw sun_glare | ordinal | 393 | 0.675 | 0.815 | 0.855 |
| étage set | nominal | 393 | 0.748 | 0.850 | 0.860 |
| étage low present | nominal | 393 | 0.460 | 0.960 | 0.966 |
| étage mid present | nominal | 393 | 1.000 | 1.000 | 1.000 |
| étage high present | nominal | 393 | 0.700 | 0.873 | 0.840 |
| genus set | nominal | 393 | 0.357 | 0.506 | 0.617 |
| genus Ci present | nominal | 393 | 0.493 | 0.767 | 0.787 |
| genus Cc present | nominal | 393 | n/a | 1.000 | 1.000 |
| genus Cs present | nominal | 393 | -0.045 | 0.715 | 0.776 |
| genus Ac present | nominal | 393 | 1.000 | 1.000 | 1.000 |
| genus As present | nominal | 393 | n/a | 1.000 | 1.000 |
| genus Ns present | nominal | 393 | n/a | 1.000 | 1.000 |
| genus Sc present | nominal | 393 | n/a | 1.000 | 1.000 |
| genus St present | nominal | 393 | n/a | 1.000 | 1.000 |
| genus Cu present | nominal | 393 | 0.488 | 0.955 | 0.964 |
| genus Cb present | nominal | 393 | 0.664 | 0.995 | 0.997 |
| cloud present (oktas > 0) | nominal | 393 | n/a | 1.000 | 1.000 |
| clear | nominal | 393 | n/a | 1.000 | 1.000 |

## Decisions

- **B0268 ceiling for the C2-style comparison (P083):** the two annotators agree on the exact étage set in 91.1% of frames (alpha 0.85) and on total cover within ±1 okta in 77.1% (interval alpha 0.81). A model's agreement with the pair's majority is compared with these numbers.
- **Genus is the weak signal again** (exact genus set alpha 0.67; high-cloud genera are where the raters differ: Ci against Ci+Cs against none on hazy days), so per-genus presence is a soft target (share of raters), never a hard label; the model is scored on the étage set, total cover and genus-in-the-raters'-union.
- **Every annotation is usable and none is obscured** (786 annotations): the four days have no rain drops on the lens; the 2026-10-21 session was paused during rain instead.
- **Targets per frame:** `data/b0268_targets.parquet` (étage and genus shares, oktas distribution, clear / contrail / glare / obstruction shares) from the protocol pair; `data/b0268_targets_all.parquet` from every rater for sensitivity; `data/b0268_derived.parquet` keeps every rater's own labels, so a model is scored against each rater and averaged, the way the ceiling is measured. With two raters a 'majority' at threshold 0.5 is the union of the pair; the hard label table uses that union as a convenience view only. Training uses the shares as distributions, as for Montenegro (P030, P035).
- **The annotators are not the two human labellers the plan names.** Both sheets of the pair were produced by language-model annotators (sheet A by Claude Fable 5, sheet B by Claude Sonnet 5.5; the further sheets C and D by Claude Fable 5.1 and Claude Opus 5.5), each from the same contact sheets, independently and without seeing another sheet. The owner's and a friend's labels were lost in a disk failure (2026-10-22). The single-annotator guard of the plan is partly met (independent raters), the human guard is not; the report says so wherever the B0268 numbers appear, and human labels on a subset remain owner items.
