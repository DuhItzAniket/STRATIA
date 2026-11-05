# Ablations (P091, validation data)

Every row is a validation number read from the Stage H result tables or the five P091 runs; the reference is the Gate G4 configuration. Global rows: mean étage macro-F1 over the sources and mean genus macro-F1 (CCSN, MGCD, SWIMCAT), the Montenegro C2 ratios and the Eye2Sky lowest-étage accuracy; one seed each, the reference's spread over three seeds is étage ± nan, genus ± nan. Backbone size was not ablated: only 224-px CLS caches exist for ViT-B/L (the P054 probes stand in); augmentation does not apply to heads trained on cached features (P080 covers regularisation). Not retrained here; the locked test (P090) was read once and is not touched.

## Global heads

| Group | Variant | Étage F1 (Δ) | Genus F1 (Δ) | C2 étage ratio | C2 oktas ratio | Eye2Sky lowest-étage acc | Run |
|---|---|---|---|---|---|---|---|
| reference | Gate G4 heads (geometry dropout 0.3, soft and set-valued targets, hierarchy loss 0.1, Eye2Sky étage rows) | 0.817 (+0.000) | 0.719 (+0.000) | - | - | - | `20261011T125754Z` |
| geometry (SkyRay) | no geometry: ray map and Sun always dropped (p_drop 1.0) | 0.809 (-0.008) | 0.720 (+0.001) | - | - | - | `20261105T134909Z` |
| geometry (SkyRay) | geometry always given (p_drop 0.0) | 0.829 (+0.011) | 0.709 (-0.010) | - | - | - | `20261105T135443Z` |
| targets | hard targets: first listed genus, majority étages, majority okta | 0.794 (-0.023) | 0.729 (+0.010) | - | - | - | `20261105T135923Z` |
| loss | no hierarchy loss (lambda_h 0) | 0.813 (-0.005) | 0.712 (-0.007) | - | - | - | `20261105T140344Z` |
| loss | uncertainty (Kendall) task weighting | 0.814 (-0.003) | 0.716 (-0.002) | - | - | - | `20261105T140931Z` |
| Eye2Sky étage rows | without the ceilometer étage rows (indomain) | 0.823 (+0.005) | 0.738 (+0.019) | - | - | - | `20261011T100505Z` |
| regularisation (P080) | head dropout 0 | 0.804 (-0.013) | 0.703 (-0.016) | - | - | - | `20261011T122932Z` |
| regularisation (P080) | head dropout 0.3 | 0.817 (-0.000) | 0.743 (+0.024) | - | - | - | `20261011T123353Z` |
| regularisation (P080) | weight decay 0 | 0.816 (-0.001) | 0.722 (+0.004) | - | - | - | `20261011T121812Z` |
| regularisation (P080) | weight decay 0.2 | 0.815 (-0.002) | 0.715 (-0.004) | - | - | - | `20261011T122312Z` |
| regularisation (P080) | no EMA | 0.820 (+0.003) | 0.711 (-0.008) | - | - | - | `20261011T123824Z` |
| regularisation (P080) | EMA 0.999 | 0.812 (-0.006) | 0.708 (-0.011) | - | - | - | `20261011T124318Z` |

| LODO fold | Variant | Held-out étage F1 |
|---|---|---|
| held-out ccsn | with Eye2Sky étage rows | 0.489 |
| held-out ccsn | without | 0.486 |
| held-out mgcd | with Eye2Sky étage rows | 0.749 |
| held-out mgcd | without | 0.871 |
| held-out montenegro | with Eye2Sky étage rows | 0.360 |
| held-out montenegro | without | 0.258 |
| held-out almeria | with Eye2Sky étage rows | 0.593 |
| held-out almeria | without | 0.327 |

## multi-task (P076)

| Variant | mean_etage_f1 | sky_miou | layer_miou | cbh_mae_m |
|---|---|---|---|---|
| joint training: joint | 0.812 | - | 0.544 | - |
| joint training: joint_long | 0.799 | - | 0.421 | - |

## partial unfreeze (P077)

| Variant | mean_etage_f1 | mean_etage_f1_before | mean_genus_f1 |
|---|---|---|---|
| LoRA: lora (after / before) | 0.810 | 0.805 | - |
| LoRA: lora_lowhead (after / before) | 0.810 | 0.805 | - |

## CBH head (P075)

| Variant | cbh_mae_m | cbh_skill | cbh_coverage_80 |
|---|---|---|---|
| with geometry | 566.100 | 0.506 | 0.917 |
| geometry zeroed | 565.821 | 0.507 | 0.944 |
| with geometry (no geometry dropout) | 550.647 | 0.520 | 0.943 |
| geometry zeroed (no geometry dropout) | 764.768 | 0.333 | 0.908 |
| with geometry (p-drop 0.15) | 560.475 | 0.511 | 0.942 |
| geometry zeroed (p-drop 0.15) | 556.864 | 0.514 | 0.946 |
| with geometry (+synthetic views) | 538.309 | 0.531 | 0.917 |
| geometry zeroed (+synthetic views) | 541.414 | 0.528 | 0.936 |
| with geometry (day-weighted) | 574.180 | 0.499 | 0.928 |
| geometry zeroed (day-weighted) | 581.758 | 0.493 | 0.944 |
| with geometry [P075 base] | 566.100 | 0.506 | 0.917 |
| geometry zeroed [P075 base] | 565.821 | 0.507 | 0.944 |
| with geometry (OLDLR only; WESTE held out) | 610.039 | 0.470 | 0.893 |
| geometry zeroed (OLDLR only; WESTE held out) | 615.340 | 0.466 | 0.915 |
| with geometry (train days 0.1) | 755.694 | 0.330 | 0.821 |
| geometry zeroed (train days 0.1) | 765.749 | 0.321 | 0.855 |
| with geometry (train days 0.25) | 661.118 | 0.420 | 0.915 |
| geometry zeroed (train days 0.25) | 672.154 | 0.411 | 0.931 |
| with geometry (train days 0.5) | 580.322 | 0.503 | 0.907 |
| geometry zeroed (train days 0.5) | 614.116 | 0.474 | 0.929 |

## dense heads (P074)

| Variant | sky_miou | layer_miou | clear_sky_cloud_fraction |
|---|---|---|---|
| dense_heads | 0.919 | - | 0.220 |
| dense_heads+layers | 0.894 | 0.559 | 0.357 |
| dense_heads+layers+kendall | 0.904 | 0.558 | 0.290 |
