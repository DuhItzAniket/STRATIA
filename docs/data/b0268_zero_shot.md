# B0268 zero-shot evaluation (P083)

Model `stratia` 1.0.0-demo (`stratia.onnx`, git `98c756df80`; the Gate G4 heads with the P089 calibration folded in) on the labelled B0268 frames of draw `k0_d0`, roles ['val'] (the locked test day, 120 frames, is not evaluated: P090). Nominal B0268 camera, zenith pose, Sun from the manifest; ONNX Runtime on the CPU, 221 ms per frame. Run `20261023T120741Z_b0268_zero_shot_6a9765a3d2`.

## Against each rater of the P041 pair (mean of A and B)

The model is scored against every rater separately and the scores are averaged, the way the ceiling is measured (one rater against the other: étage set 0.911, oktas within ±1 0.771); `c2_*_ratio` = model / raters, as in criterion C2. No tie between the two raters is broken anywhere in this table.

| Day | n | n_raters | etage_set_agreement | etage_set_agreement_A | etage_set_agreement_B | c2_etage_ratio | oktas_within1 | oktas_within1_A | oktas_within1_B | c2_oktas_ratio | oktas_mae | oktas_bias | genus_top1_in_set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 273.000 | 2.000 | 0.031 | 0.033 | 0.029 | 0.034 | 0.278 | 0.245 | 0.311 | 0.361 | 2.608 | 2.396 | 0.300 |
| 2026-10-18 | 120.000 | 2.000 | 0.046 | 0.050 | 0.042 | 0.050 | 0.492 | 0.492 | 0.492 | 0.638 | 1.617 | 1.133 | 0.633 |
| 2026-10-19 | 120.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.075 | 0.000 | 0.150 | 0.097 | 3.142 | 3.142 | 0.008 |
| 2026-10-21 | 33.000 | 2.000 | 0.091 | 0.091 | 0.091 | 0.100 | 0.242 | 0.242 | 0.242 | 0.314 | 4.273 | 4.273 | 0.152 |

## Against the pair's majority and mean cover (cloud fraction, glare, OOD)

Truth = étages named by at least one of the two raters, the lower okta on their ties, the union of their genera; cover = the raters' mean okta / 8. Use this table for the sky-parse, glare and OOD columns; the étage and okta columns here are the union / lower-okta view, the table above is the honest one.

| Day | n | etage_set_agreement | etage_f1 | oktas_within1 | genus_top1_in_union | cloud_fraction_mae_vs_cover | cloud_fraction_corr_vs_cover | cloud_fraction_on_le1_okta | glare_fraction_strong_frames | glare_fraction_other_frames | ood_mean | ood_flagged_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 273.000 | 0.022 | 0.514 | 0.245 | 0.300 | 0.107 | 0.833 | 0.360 | 0.000 | 0.000 | 1.127 | 0.084 |
| 2026-10-18 | 120.000 | 0.025 | 0.409 | 0.492 | 0.633 | 0.115 | -0.079 | - | 0.000 | 0.000 | 0.949 | 0.000 |
| 2026-10-19 | 120.000 | 0.000 | 0.476 | 0.000 | 0.008 | 0.074 | 0.787 | - | 0.000 | - | 1.176 | 0.017 |
| 2026-10-21 | 33.000 | 0.091 | 0.447 | 0.242 | 0.152 | 0.200 | 0.830 | 0.360 | 0.000 | - | 1.598 | 0.636 |

## Sensitivity: against each of the four raters (A, B, C, D), averaged

| Day | n | n_raters | etage_set_agreement | c2_etage_ratio | oktas_within1 | c2_oktas_ratio | oktas_mae | oktas_bias | genus_top1_in_set |
|---|---|---|---|---|---|---|---|---|---|
| all | 273.000 | 4.000 | 0.037 | 0.043 | 0.291 | 0.344 | 2.532 | 2.332 | 0.333 |
| 2026-10-18 | 120.000 | 4.000 | 0.058 | 0.068 | 0.477 | 0.563 | 1.604 | 1.150 | 0.633 |
| 2026-10-19 | 120.000 | 4.000 | 0.000 | 0.000 | 0.119 | 0.140 | 2.977 | 2.977 | 0.083 |
| 2026-10-21 | 33.000 | 4.000 | 0.091 | 0.106 | 0.242 | 0.286 | 4.288 | 4.288 | 0.152 |

Columns: `etage_set_agreement` = exact étage set (sigmoid > 0.5 per étage) against a rater's set; `oktas_within1` = argmax okta within ±1 of a rater's okta; `genus_top1_in_set` = the top genus is one that rater named (frames without a named genus skipped); `cloud_fraction_*` = sky-parse cloud share of the valid sky against the raters' mean cover; `cloud_fraction_on_le1_okta` = that share on frames the raters put at 0-1 okta (the research design's no-collapse check asks for < 0.20 on clear frames); `glare_fraction_*` = sky-parse glare share on frames the raters called strong glare and on the others; `ood_flagged_share` = frames above the model card's OOD threshold.
