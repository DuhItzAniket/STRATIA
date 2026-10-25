# Locked-test evaluation (P090, validation check)

Split `val`; lock `c41841f43cc488d7`; run `20261025T113026Z_locked_test_val_0f1104621f`; git `9a18b7e`; 500 cluster-bootstrap resamples; models and protocol as pre-registered in `docs/phases/P090-locked-test.md`. Thresholds 0.5; one seed except the in-domain global heads (three).

## The Gate G4 criteria on these rows

| Criterion | Quantity | P082 (val) | This split | 95 % CI | Reference | Required | Holds |
|---|---|---|---|---|---|---|---|
| C1 | LODO étage macro-F1, mean of the four folds | 0.548 | 0.548 |  | 0.582 (probe ViT-B/16 @ 224, refitted) | ≥ reference + 0.05 | no |
| C1 | in-domain genus macro-F1 (CCSN, MGCD) | 0.719 | 0.719 |  | 0.716 (probe ViT-L/16 @ 224, refitted) | ≥ reference − 0.01 | yes |
| C2 | Montenegro étage-set agreement / human ceiling | 1.047 | 1.047 | [0.929, 1.141] | 0.900 (pre-registered bar) | ≥ 0.90 | yes |
| C2 | Montenegro oktas ±1 agreement / human ceiling | 1.027 | 1.027 | [0.998, 1.049] | 0.900 (pre-registered bar) | ≥ 0.90 | yes |
| C3 | CBH skill, OLDLR (CDLRA) pairs | 0.471 | 0.471 | [0.299, 0.661] | 0.200 (pre-registered bar) | ≥ 0.20 | yes |
| C3 | CBH skill, WESTE (CDLRB) pairs, OLDLR-only training | 0.448 | 0.448 | [0.198, 0.621] | 0.100 (pre-registered bar) | ≥ 0.10 | yes |
| C3 | CBH MAE (m), all pairs, against the frozen-feature ridge | 566 | 566 | [386, 744] | 699 (ridge DINOv3 ViT-S/16 @ 512, refitted (the P058 CNN kept no checkpoint; 724 m on val)) | ≤ 0.9 × reference | yes |
| C3 | 80 % interval coverage with the P089 temperature and radius | 0.802 | 0.801 |  | 0.800 (nominal) | 0.75–0.85 | yes |

## Every number

### `in_domain`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| almeria | `indomain_e2s_s0` | etage_f1 | 0.775 | [0.601, 0.841] | 84 |  |
| almeria | `indomain_e2s_s0` | etage_unsupported_fp | - |  | 84 |  |
| almeria | `indomain_e2s_s0` | etage_f1_tuned | 0.813 | [0.745, 0.876] | 84 |  |
| almeria | `indomain_e2s_s0` | n_etage | 84.000 | [52.000, 120] | 84 |  |
| ccsn | `indomain_e2s_s0` | etage_f1 | 0.802 | [0.752, 0.850] | 255 |  |
| ccsn | `indomain_e2s_s0` | etage_unsupported_fp | - |  | 255 |  |
| ccsn | `indomain_e2s_s0` | etage_f1_tuned | 0.810 | [0.775, 0.860] | 255 |  |
| ccsn | `indomain_e2s_s0` | n_etage | 236 | [228, 244] | 255 |  |
| ccsn | `indomain_e2s_s0` | genus_f1 | 0.626 | [0.563, 0.680] | 255 |  |
| ccsn | `indomain_e2s_s0` | genus_acc | 0.643 | [0.588, 0.705] | 255 |  |
| ccsn | `indomain_e2s_s0` | n_genus | 255 | [246, 265] | 255 |  |
| mgcd | `indomain_e2s_s0` | etage_f1 | 1.000 | [1.000, 1.000] | 959 |  |
| mgcd | `indomain_e2s_s0` | etage_unsupported_fp | 0.000 | [0.000, 0.000] | 959 |  |
| mgcd | `indomain_e2s_s0` | etage_f1_tuned | 1.000 | [1.000, 1.000] | 959 |  |
| mgcd | `indomain_e2s_s0` | n_etage | 438 | [100, 877] | 959 |  |
| mgcd | `indomain_e2s_s0` | genus_f1 | 0.812 | [0.624, 1.000] | 959 |  |
| mgcd | `indomain_e2s_s0` | genus_acc | 0.857 | [0.697, 1.000] | 959 |  |
| mgcd | `indomain_e2s_s0` | n_genus | 789 | [383, 1214] | 959 |  |
| montenegro | `indomain_e2s_s0` | etage_f1 | 0.692 | [0.605, 0.988] | 511 |  |
| montenegro | `indomain_e2s_s0` | etage_unsupported_fp | - | [0.005, 0.005] | 511 |  |
| montenegro | `indomain_e2s_s0` | etage_f1_tuned | 0.708 | [0.624, 0.991] | 511 |  |
| montenegro | `indomain_e2s_s0` | n_etage | 511 | [450, 572] | 511 |  |
| montenegro | `indomain_e2s_s0` | oktas_mae | 0.489 | [0.479, 0.502] | 511 |  |
| montenegro | `indomain_e2s_s0` | oktas_within1 | 0.896 | [0.871, 0.916] | 511 |  |
| montenegro | `indomain_e2s_s0` | oktas_kl | 0.573 | [0.517, 0.644] | 511 |  |
| montenegro | `indomain_e2s_s0` | etage_set_agreement | 0.892 | [0.791, 0.972] | 511 |  |
| montenegro | `indomain_e2s_s0` | c2_etage_ratio | 1.047 | [0.929, 1.141] | 511 |  |
| montenegro | `indomain_e2s_s0` | c2_oktas_ratio | 1.027 | [0.998, 1.049] | 511 |  |
| almeria | `indomain_e2s_s1` | etage_f1 | 0.773 |  | 84 |  |
| almeria | `indomain_e2s_s1` | etage_unsupported_fp | - |  | 84 |  |
| almeria | `indomain_e2s_s1` | etage_f1_tuned | 0.818 |  | 84 |  |
| almeria | `indomain_e2s_s1` | n_etage | 84.000 |  | 84 |  |
| ccsn | `indomain_e2s_s1` | etage_f1 | 0.761 |  | 255 |  |
| ccsn | `indomain_e2s_s1` | etage_unsupported_fp | - |  | 255 |  |
| ccsn | `indomain_e2s_s1` | etage_f1_tuned | 0.802 |  | 255 |  |
| ccsn | `indomain_e2s_s1` | n_etage | 236 |  | 255 |  |
| ccsn | `indomain_e2s_s1` | genus_f1 | 0.643 |  | 255 |  |
| ccsn | `indomain_e2s_s1` | genus_acc | 0.651 |  | 255 |  |
| ccsn | `indomain_e2s_s1` | n_genus | 255 |  | 255 |  |
| mgcd | `indomain_e2s_s1` | etage_f1 | 1.000 |  | 959 |  |
| mgcd | `indomain_e2s_s1` | etage_unsupported_fp | 0.000 |  | 959 |  |
| mgcd | `indomain_e2s_s1` | etage_f1_tuned | 1.000 |  | 959 |  |
| mgcd | `indomain_e2s_s1` | n_etage | 438 |  | 959 |  |
| mgcd | `indomain_e2s_s1` | genus_f1 | 0.833 |  | 959 |  |
| mgcd | `indomain_e2s_s1` | genus_acc | 0.868 |  | 959 |  |
| mgcd | `indomain_e2s_s1` | n_genus | 789 |  | 959 |  |
| montenegro | `indomain_e2s_s1` | etage_f1 | 0.733 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | etage_unsupported_fp | - |  | 511 |  |
| montenegro | `indomain_e2s_s1` | etage_f1_tuned | 0.752 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | n_etage | 511 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | oktas_mae | 0.460 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | oktas_within1 | 0.908 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | oktas_kl | 0.561 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | etage_set_agreement | 0.888 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | c2_etage_ratio | 1.043 |  | 511 |  |
| montenegro | `indomain_e2s_s1` | c2_oktas_ratio | 1.040 |  | 511 |  |
| almeria | `indomain_e2s_s2` | etage_f1 | 0.757 |  | 84 |  |
| almeria | `indomain_e2s_s2` | etage_unsupported_fp | - |  | 84 |  |
| almeria | `indomain_e2s_s2` | etage_f1_tuned | 0.809 |  | 84 |  |
| almeria | `indomain_e2s_s2` | n_etage | 84.000 |  | 84 |  |
| ccsn | `indomain_e2s_s2` | etage_f1 | 0.783 |  | 255 |  |
| ccsn | `indomain_e2s_s2` | etage_unsupported_fp | - |  | 255 |  |
| ccsn | `indomain_e2s_s2` | etage_f1_tuned | 0.795 |  | 255 |  |
| ccsn | `indomain_e2s_s2` | n_etage | 236 |  | 255 |  |
| ccsn | `indomain_e2s_s2` | genus_f1 | 0.616 |  | 255 |  |
| ccsn | `indomain_e2s_s2` | genus_acc | 0.624 |  | 255 |  |
| ccsn | `indomain_e2s_s2` | n_genus | 255 |  | 255 |  |
| mgcd | `indomain_e2s_s2` | etage_f1 | 1.000 |  | 959 |  |
| mgcd | `indomain_e2s_s2` | etage_unsupported_fp | 0.000 |  | 959 |  |
| mgcd | `indomain_e2s_s2` | etage_f1_tuned | 1.000 |  | 959 |  |
| mgcd | `indomain_e2s_s2` | n_etage | 438 |  | 959 |  |
| mgcd | `indomain_e2s_s2` | genus_f1 | 0.817 |  | 959 |  |
| mgcd | `indomain_e2s_s2` | genus_acc | 0.856 |  | 959 |  |
| mgcd | `indomain_e2s_s2` | n_genus | 789 |  | 959 |  |
| montenegro | `indomain_e2s_s2` | etage_f1 | 0.675 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | etage_unsupported_fp | - |  | 511 |  |
| montenegro | `indomain_e2s_s2` | etage_f1_tuned | 0.683 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | n_etage | 511 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | oktas_mae | 0.456 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | oktas_within1 | 0.906 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | oktas_kl | 0.568 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | etage_set_agreement | 0.888 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | c2_etage_ratio | 1.043 |  | 511 |  |
| montenegro | `indomain_e2s_s2` | c2_oktas_ratio | 1.038 |  | 511 |  |
| almeria | `indomain_e2s_mean3` | etage_f1 | 0.768 |  |  | std over 3 seeds 0.0081 |
| almeria | `indomain_e2s_mean3` | etage_f1_tuned | 0.814 |  |  | std over 3 seeds 0.0037 |
| almeria | `indomain_e2s_mean3` | n_etage | 84.000 |  |  | std over 3 seeds 0.0000 |
| ccsn | `indomain_e2s_mean3` | etage_f1 | 0.782 |  |  | std over 3 seeds 0.0167 |
| ccsn | `indomain_e2s_mean3` | etage_f1_tuned | 0.803 |  |  | std over 3 seeds 0.0060 |
| ccsn | `indomain_e2s_mean3` | genus_acc | 0.639 |  |  | std over 3 seeds 0.0115 |
| ccsn | `indomain_e2s_mean3` | genus_f1 | 0.628 |  |  | std over 3 seeds 0.0112 |
| ccsn | `indomain_e2s_mean3` | n_etage | 236 |  |  | std over 3 seeds 0.0000 |
| ccsn | `indomain_e2s_mean3` | n_genus | 255 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | etage_f1 | 1.000 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | etage_f1_tuned | 1.000 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | etage_unsupported_fp | 0.000 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | genus_acc | 0.860 |  |  | std over 3 seeds 0.0057 |
| mgcd | `indomain_e2s_mean3` | genus_f1 | 0.820 |  |  | std over 3 seeds 0.0091 |
| mgcd | `indomain_e2s_mean3` | n_etage | 438 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | n_genus | 789 |  |  | std over 3 seeds 0.0000 |
| montenegro | `indomain_e2s_mean3` | c2_etage_ratio | 1.044 |  |  | std over 3 seeds 0.0022 |
| montenegro | `indomain_e2s_mean3` | c2_oktas_ratio | 1.035 |  |  | std over 3 seeds 0.0059 |
| montenegro | `indomain_e2s_mean3` | etage_f1 | 0.700 |  |  | std over 3 seeds 0.0243 |
| montenegro | `indomain_e2s_mean3` | etage_f1_tuned | 0.714 |  |  | std over 3 seeds 0.0285 |
| montenegro | `indomain_e2s_mean3` | etage_set_agreement | 0.890 |  |  | std over 3 seeds 0.0018 |
| montenegro | `indomain_e2s_mean3` | n_etage | 511 |  |  | std over 3 seeds 0.0000 |
| montenegro | `indomain_e2s_mean3` | oktas_kl | 0.568 |  |  | std over 3 seeds 0.0048 |
| montenegro | `indomain_e2s_mean3` | oktas_mae | 0.468 |  |  | std over 3 seeds 0.0148 |
| montenegro | `indomain_e2s_mean3` | oktas_within1 | 0.903 |  |  | std over 3 seeds 0.0051 |
| ccsn | `probe_dinov3_vitl16_224` | genus_f1 | 0.646 | [0.581, 0.694] | 255 | C 0.01 chosen on val |
| ccsn | `probe_dinov3_vitl16_224` | genus_acc | 0.659 |  | 255 |  |
| mgcd | `probe_dinov3_vitl16_224` | genus_f1 | 0.787 | [0.440, 0.868] | 959 | C 0.01 chosen on val |
| mgcd | `probe_dinov3_vitl16_224` | genus_acc | 0.814 |  | 959 |  |

### `cbh_pairs_etage`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| eye2sky_cbh | `indomain_e2s_s0` | etage_recall | 0.972 | [0.939, 0.992] | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s0` | lowest_etage_acc | 0.788 | [0.654, 0.907] | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s0` | n_etage | 1233 | [874, 1602] | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s1` | etage_recall | 0.968 |  | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s1` | lowest_etage_acc | 0.805 |  | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s1` | n_etage | 1233 |  | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s2` | etage_recall | 0.972 |  | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s2` | lowest_etage_acc | 0.815 |  | 1233 |  |
| eye2sky_cbh | `indomain_e2s_s2` | n_etage | 1233 |  | 1233 |  |
| eye2sky_cbh | `indomain_e2s_mean3` | etage_recall | 0.971 |  |  | std over 3 seeds 0.0023 |
| eye2sky_cbh | `indomain_e2s_mean3` | lowest_etage_acc | 0.803 |  |  | std over 3 seeds 0.0110 |
| eye2sky_cbh | `indomain_e2s_mean3` | n_etage | 1233 |  |  | std over 3 seeds 0.0000 |

### `ood`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| in_domain_all | `indomain_e2s_s0` | flagged_share | 0.050 |  | 3121 |  |
| almeria | `indomain_e2s_s0` | flagged_share | 0.060 |  | 84 |  |
| ccsn | `indomain_e2s_s0` | flagged_share | 0.443 |  | 255 |  |
| eye2sky | `indomain_e2s_s0` | flagged_share | 0.000 |  | 1233 |  |
| mgcd | `indomain_e2s_s0` | flagged_share | 0.022 |  | 959 |  |
| montenegro | `indomain_e2s_s0` | flagged_share | 0.002 |  | 511 |  |
| swim | `indomain_e2s_s0` | flagged_share | 0.203 |  | 79 |  |
| b0268 | `indomain_e2s_s0` | flagged_share | 0.077 |  | 273 |  |
| b0268_vs_in_domain | `indomain_e2s_s0` | auroc | 0.901 |  | 3394 | higher score = out of domain; cached-feature path for both |

### `lodo`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| ccsn | `lodo_ccsn_e2s` | etage_f1 | 0.489 | [0.431, 0.543] | 255 |  |
| ccsn | `lodo_ccsn_e2s` | etage_unsupported_fp | - |  | 255 |  |
| ccsn | `lodo_ccsn_e2s` | etage_f1_tuned | 0.565 | [0.525, 0.618] | 255 |  |
| ccsn | `lodo_ccsn_e2s` | n_etage | 236 | [228, 244] | 255 |  |
| ccsn | `lodo_ccsn_e2s` | genus_f1 | 0.095 | [0.066, 0.120] | 255 |  |
| ccsn | `lodo_ccsn_e2s` | genus_acc | 0.165 | [0.122, 0.209] | 255 |  |
| ccsn | `lodo_ccsn_e2s` | n_genus | 255 | [246, 265] | 255 |  |
| ccsn | `lodo_ccsn_e2s` | etage_f1_valtuned | 0.194 |  | 236 | thresholds chosen on the fold's val rows (= the evaluated rows) |
| mgcd | `lodo_mgcd_e2s` | etage_f1 | 0.749 | [0.494, 0.995] | 959 |  |
| mgcd | `lodo_mgcd_e2s` | etage_unsupported_fp | 0.039 | [0.003, 0.503] | 959 |  |
| mgcd | `lodo_mgcd_e2s` | etage_f1_tuned | 0.946 | [0.569, 1.000] | 959 |  |
| mgcd | `lodo_mgcd_e2s` | n_etage | 438 | [100, 877] | 959 |  |
| mgcd | `lodo_mgcd_e2s` | genus_f1 | 0.401 | [0.172, 0.746] | 959 |  |
| mgcd | `lodo_mgcd_e2s` | genus_acc | 0.408 | [0.118, 0.836] | 959 |  |
| mgcd | `lodo_mgcd_e2s` | n_genus | 789 | [383, 1214] | 959 |  |
| mgcd | `lodo_mgcd_e2s` | etage_f1_valtuned | 0.000 |  | 438 | thresholds chosen on the fold's val rows (= the evaluated rows) |
| montenegro | `lodo_montenegro_e2s` | etage_f1 | 0.360 | [0.360, 0.796] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | etage_unsupported_fp | - | [0.460, 0.460] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | etage_f1_tuned | 0.462 | [0.457, 0.978] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | n_etage | 511 | [450, 572] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | oktas_mae | 4.536 | [3.521, 5.827] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | oktas_within1 | 0.346 | [0.160, 0.493] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | oktas_kl | 1.467 | [1.402, 1.549] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | etage_set_agreement | 0.245 | [0.073, 0.462] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | c2_etage_ratio | 0.287 | [0.086, 0.543] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | c2_oktas_ratio | 0.397 | [0.183, 0.565] | 511 |  |
| montenegro | `lodo_montenegro_e2s` | etage_f1_valtuned | 0.077 |  | 511 | thresholds chosen on the fold's val rows (= the evaluated rows) |
| almeria | `lodo_almeria_e2s` | etage_f1 | 0.593 | [0.394, 0.708] | 84 |  |
| almeria | `lodo_almeria_e2s` | etage_unsupported_fp | - |  | 84 |  |
| almeria | `lodo_almeria_e2s` | etage_f1_tuned | 0.630 | [0.544, 0.750] | 84 |  |
| almeria | `lodo_almeria_e2s` | n_etage | 84.000 | [52.000, 120] | 84 |  |
| almeria | `lodo_almeria_e2s` | etage_f1_valtuned | 0.171 |  | 84 | thresholds chosen on the fold's val rows (= the evaluated rows) |
| ccsn | `probe_dinov3_vitb16_224` | etage_f1 | 0.390 | [0.330, 0.447] | 236 | C 0.1 chosen on val |
| ccsn | `probe_dinov3_vitb16_224` | etage_unsupported_fp | - |  | 236 |  |
| mgcd | `probe_dinov3_vitb16_224` | etage_f1 | 0.995 | [0.893, 1.000] | 438 | C 0.01 chosen on val |
| mgcd | `probe_dinov3_vitb16_224` | etage_unsupported_fp | 0.000 |  | 438 |  |
| montenegro | `probe_dinov3_vitb16_224` | etage_f1 | 0.517 | [0.369, 0.517] | 511 | C 0.01 chosen on val |
| montenegro | `probe_dinov3_vitb16_224` | etage_unsupported_fp | - |  | 511 |  |
| almeria | `probe_dinov3_vitb16_224` | etage_f1 | 0.426 | [0.219, 0.526] | 84 | C 0.1 chosen on val |
| almeria | `probe_dinov3_vitb16_224` | etage_unsupported_fp | - |  | 84 |  |

### `dense`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| all | `dense_sky_layer_s0` | sky_miou | 0.904 | [0.900, 0.909] | 891 |  |
| all | `dense_sky_layer_s0` | iou_invalid | 0.964 |  | 891 |  |
| all | `dense_sky_layer_s0` | iou_sky | 0.869 |  | 891 |  |
| all | `dense_sky_layer_s0` | iou_cloud | 0.879 |  | 891 |  |
| all | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.503 |  | 891 |  |
| all | `dense_sky_layer_s0` | layer_miou | 0.558 | [0.471, 0.651] | 891 |  |
| all | `dense_sky_layer_s0` | layer_iou_low | 0.577 |  | 891 |  |
| all | `dense_sky_layer_s0` | layer_iou_mid | 0.341 |  | 891 |  |
| all | `dense_sky_layer_s0` | layer_iou_high | 0.757 |  | 891 |  |
| almeria | `dense_sky_layer_s0` | sky_miou | 0.885 | [0.869, 0.902] | 84 |  |
| almeria | `dense_sky_layer_s0` | iou_invalid | 0.964 |  | 84 |  |
| almeria | `dense_sky_layer_s0` | iou_sky | 0.868 |  | 84 |  |
| almeria | `dense_sky_layer_s0` | iou_cloud | 0.823 |  | 84 |  |
| almeria | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.330 |  | 84 |  |
| almeria | `dense_sky_layer_s0` | layer_miou | 0.558 | [0.466, 0.664] | 84 |  |
| almeria | `dense_sky_layer_s0` | layer_iou_low | 0.577 |  | 84 |  |
| almeria | `dense_sky_layer_s0` | layer_iou_mid | 0.341 |  | 84 |  |
| almeria | `dense_sky_layer_s0` | layer_iou_high | 0.757 |  | 84 |  |
| shwimseg | `dense_sky_layer_s0` | sky_miou | 0.809 | [0.755, 0.855] | 16 |  |
| shwimseg | `dense_sky_layer_s0` | iou_sky | 0.779 |  | 16 |  |
| shwimseg | `dense_sky_layer_s0` | iou_cloud | 0.839 |  | 16 |  |
| shwimseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.578 |  | 16 |  |
| swimseg | `dense_sky_layer_s0` | sky_miou | 0.878 | [0.860, 0.897] | 102 |  |
| swimseg | `dense_sky_layer_s0` | iou_sky | 0.874 |  | 102 |  |
| swimseg | `dense_sky_layer_s0` | iou_cloud | 0.881 |  | 102 |  |
| swimseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.506 |  | 102 |  |
| swinseg | `dense_sky_layer_s0` | sky_miou | 0.903 | [0.884, 0.921] | 12 |  |
| swinseg | `dense_sky_layer_s0` | iou_sky | 0.911 |  | 12 |  |
| swinseg | `dense_sky_layer_s0` | iou_cloud | 0.895 |  | 12 |  |
| swinseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.450 |  | 12 |  |
| swinyseg | `dense_sky_layer_s0` | sky_miou | 0.877 | [0.869, 0.885] | 677 |  |
| swinyseg | `dense_sky_layer_s0` | iou_sky | 0.870 |  | 677 |  |
| swinyseg | `dense_sky_layer_s0` | iou_cloud | 0.884 |  | 677 |  |
| swinyseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.524 |  | 677 |  |
| clear_sky_val | `dense_sky_layer_s0` | fraction_invalid | 0.000 |  | 200 |  |
| clear_sky_val | `dense_sky_layer_s0` | fraction_sky | 0.718 |  | 200 |  |
| clear_sky_val | `dense_sky_layer_s0` | fraction_cloud | 0.282 |  | 200 |  |
| clear_sky_val | `dense_sky_layer_s0` | fraction_sun_glare | 0.000 |  | 200 |  |

### `cbh_pairs`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| all | `climatology_train` | mae_m | 1147 |  | 1368 | clim 1726 m |
| all | `cbh_base_s0` | n | 1368 |  | 2193 |  |
| all | `cbh_base_s0` | mae_m | 566 | [386, 744] | 2193 |  |
| all | `cbh_base_s0` | bias_m | 125 |  | 2193 |  |
| all | `cbh_base_s0` | rmsd_m | 1054 |  | 2193 |  |
| all | `cbh_base_s0` | skill | 0.506 | [0.351, 0.663] | 2193 |  |
| all | `cbh_base_s0` | within_250m | 0.489 |  | 2193 |  |
| all | `cbh_base_s0` | within_500m | 0.693 | [0.587, 0.804] | 2193 |  |
| all | `cbh_base_s0` | within_1000m | 0.839 |  | 2193 |  |
| all | `cbh_base_s0` | mae_0-1000 | 263 |  | 2193 |  |
| all | `cbh_base_s0` | mae_1000-2000 | 433 |  | 2193 |  |
| all | `cbh_base_s0` | mae_2000-4000 | 772 |  | 2193 |  |
| all | `cbh_base_s0` | mae_4000-8000 | 1451 |  | 2193 |  |
| all | `cbh_base_s0` | mae_8000-20000 | 1208 |  | 2193 |  |
| all | `cbh_base_s0` | crps | 0.027 | [0.020, 0.035] | 2193 |  |
| all | `cbh_base_s0` | coverage_80 | 0.917 | [0.875, 0.952] | 2193 |  |
| all | `cbh_base_s0` | cloud_balanced_acc | 0.926 |  | 2193 |  |
| all | `cbh_base_s0` | top_bin_share_pred | 0.248 |  | 2193 |  |
| all | `cbh_base_s0` | top_bin_share_true | 0.190 |  | 2193 |  |
| all | `cbh_base_s0` | entropy_ratio | 0.946 |  | 2193 |  |
| all | `cbh_base_s0` | n_cloudy | 1368 |  | 2193 |  |
| all | `cbh_base_s0` | coverage_80_calibrated | 0.801 |  | 1368 | P089 temperature 1.2394 and radius 0.3265 |
| all | `cbh_base_s0` | coverage_95_calibrated | 0.950 |  | 1368 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0` | n | 662 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_m | 607 | [388, 804] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | bias_m | 187 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | rmsd_m | 1180 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | skill | 0.471 | [0.299, 0.661] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | within_250m | 0.483 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | within_500m | 0.687 | [0.572, 0.812] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | within_1000m | 0.825 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_0-1000 | 311 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_1000-2000 | 494 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_2000-4000 | 736 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_4000-8000 | 1339 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_8000-20000 | 1342 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | crps | 0.028 | [0.020, 0.036] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | coverage_80 | 0.921 | [0.892, 0.951] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | cloud_balanced_acc | 0.925 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | top_bin_share_pred | 0.205 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | top_bin_share_true | 0.193 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | entropy_ratio | 0.963 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | n_cloudy | 662 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | coverage_80_calibrated | 0.801 |  | 662 | P089 temperature 1.2394 and radius 0.3265 |
| OLDLR | `cbh_base_s0` | coverage_95_calibrated | 0.943 |  | 662 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0` | skill_station_climatology | 0.507 |  | 1079 | secondary: against the climatology's MAE on this station's rows (1231 m) |
| WESTE | `cbh_base_s0` | n | 706 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_m | 528 | [359, 741] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | bias_m | 66.963 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | rmsd_m | 921 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | skill | 0.540 | [0.354, 0.687] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | within_250m | 0.494 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | within_500m | 0.698 | [0.585, 0.803] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | within_1000m | 0.853 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_0-1000 | 222 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_1000-2000 | 379 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_2000-4000 | 808 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_4000-8000 | 1585 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_8000-20000 | 1023 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | crps | 0.027 | [0.019, 0.035] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | coverage_80 | 0.912 | [0.860, 0.956] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | cloud_balanced_acc | 0.927 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | top_bin_share_pred | 0.299 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | top_bin_share_true | 0.187 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | entropy_ratio | 0.915 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | n_cloudy | 706 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | coverage_80_calibrated | 0.802 |  | 706 | P089 temperature 1.2394 and radius 0.3265 |
| WESTE | `cbh_base_s0` | coverage_95_calibrated | 0.958 |  | 706 | P089 temperature 1.2394 and radius 0.4376 |
| WESTE | `cbh_base_s0` | skill_station_climatology | 0.506 |  | 1114 | secondary: against the climatology's MAE on this station's rows (1068 m) |
| all | `cbh_base_s0_geometry_zeroed` | n | 1368 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_m | 566 | [385, 747] | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | bias_m | 125 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | rmsd_m | 1072 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | skill | 0.507 | [0.348, 0.664] | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | within_250m | 0.493 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | within_500m | 0.699 | [0.584, 0.818] | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | within_1000m | 0.846 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_0-1000 | 283 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_1000-2000 | 418 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_2000-4000 | 773 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_4000-8000 | 1545 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_8000-20000 | 980 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | crps | 0.027 | [0.020, 0.035] | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | coverage_80 | 0.944 | [0.910, 0.969] | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | cloud_balanced_acc | 0.921 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | top_bin_share_pred | 0.259 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | top_bin_share_true | 0.190 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | entropy_ratio | 0.911 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | n_cloudy | 1368 |  | 2193 |  |
| all | `cbh_base_s0_geometry_zeroed` | coverage_80_calibrated | 0.843 |  | 1368 | P089 temperature 1.2394 and radius 0.3265 |
| all | `cbh_base_s0_geometry_zeroed` | coverage_95_calibrated | 0.971 |  | 1368 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0_geometry_zeroed` | n | 662 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_m | 603 | [386, 798] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | bias_m | 206 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | rmsd_m | 1186 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | skill | 0.474 | [0.304, 0.664] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | within_250m | 0.482 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | within_500m | 0.696 | [0.573, 0.827] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | within_1000m | 0.832 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_0-1000 | 352 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_1000-2000 | 478 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_2000-4000 | 719 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_4000-8000 | 1436 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_8000-20000 | 1026 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | crps | 0.028 | [0.020, 0.036] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | coverage_80 | 0.938 | [0.910, 0.961] | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | cloud_balanced_acc | 0.915 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | top_bin_share_pred | 0.218 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | top_bin_share_true | 0.193 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | entropy_ratio | 0.924 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | n_cloudy | 662 |  | 1079 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | coverage_80_calibrated | 0.832 |  | 662 | P089 temperature 1.2394 and radius 0.3265 |
| OLDLR | `cbh_base_s0_geometry_zeroed` | coverage_95_calibrated | 0.964 |  | 662 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0_geometry_zeroed` | skill_station_climatology | 0.510 |  | 1079 | secondary: against the climatology's MAE on this station's rows (1231 m) |
| WESTE | `cbh_base_s0_geometry_zeroed` | n | 706 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_m | 531 | [360, 745] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | bias_m | 48.829 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | rmsd_m | 954 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | skill | 0.537 | [0.351, 0.686] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | within_250m | 0.503 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | within_500m | 0.701 | [0.577, 0.811] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | within_1000m | 0.858 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_0-1000 | 222 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_1000-2000 | 367 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_2000-4000 | 827 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_4000-8000 | 1677 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_8000-20000 | 915 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | crps | 0.027 | [0.019, 0.035] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | coverage_80 | 0.949 | [0.912, 0.978] | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | cloud_balanced_acc | 0.927 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | top_bin_share_pred | 0.306 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | top_bin_share_true | 0.187 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | entropy_ratio | 0.883 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | n_cloudy | 706 |  | 1114 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | coverage_80_calibrated | 0.853 |  | 706 | P089 temperature 1.2394 and radius 0.3265 |
| WESTE | `cbh_base_s0_geometry_zeroed` | coverage_95_calibrated | 0.977 |  | 706 | P089 temperature 1.2394 and radius 0.4376 |
| WESTE | `cbh_base_s0_geometry_zeroed` | skill_station_climatology | 0.503 |  | 1114 | secondary: against the climatology's MAE on this station's rows (1068 m) |
| WESTE | `cbh_oldlr_only_s0` | n | 706 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_m | 636 | [437, 923] | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | bias_m | 8.511 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | rmsd_m | 1019 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | skill | 0.448 | [0.198, 0.621] | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | within_250m | 0.364 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | within_500m | 0.632 | [0.518, 0.745] | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | within_1000m | 0.823 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_0-1000 | 329 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_1000-2000 | 499 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_2000-4000 | 749 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_4000-8000 | 1411 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_8000-20000 | 3203 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | crps | 0.034 | [0.024, 0.046] | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | coverage_80 | 0.861 | [0.787, 0.926] | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | cloud_balanced_acc | 0.795 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | top_bin_share_pred | 0.237 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | top_bin_share_true | 0.187 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | entropy_ratio | 0.899 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | n_cloudy | 706 |  | 1114 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | coverage_80_calibrated | 0.701 |  | 706 | P089 temperature 1.2394 and radius 0.3265 |
| WESTE | `cbh_oldlr_only_s0` | coverage_95_calibrated | 0.915 |  | 706 | P089 temperature 1.2394 and radius 0.4376 |
| WESTE | `cbh_oldlr_only_s0` | skill_station_climatology | 0.408 |  | 1114 | secondary: against the OLDLR climatology's MAE on the WESTE rows (1074 m) |
| all | `ridge_dinov3_vits16_512` | n | 1368 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_m | 699 | [487, 928] | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | bias_m | 34.021 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | rmsd_m | 1141 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | skill | 0.390 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | within_250m | 0.366 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | within_500m | 0.605 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | within_1000m | 0.781 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_0-1000 | 303 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_1000-2000 | 486 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_2000-4000 | 853 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_4000-8000 | 1859 |  | 1368 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_8000-20000 | 2903 |  | 1368 | alpha 1000.0 chosen on val |
| OLDLR | `ridge_dinov3_vits16_512` | n | 662 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_m | 718 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | bias_m | 50.040 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | rmsd_m | 1153 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | skill | 0.417 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | within_250m | 0.376 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | within_500m | 0.595 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | within_1000m | 0.775 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_0-1000 | 345 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_1000-2000 | 496 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_2000-4000 | 762 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_4000-8000 | 1738 |  | 662 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_8000-20000 | 2999 |  | 662 |  |
| WESTE | `ridge_dinov3_vits16_512` | n | 706 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_m | 682 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | bias_m | 19.001 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | rmsd_m | 1129 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | skill | 0.361 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | within_250m | 0.357 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | within_500m | 0.615 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | within_1000m | 0.788 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_0-1000 | 267 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_1000-2000 | 477 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_2000-4000 | 946 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_4000-8000 | 2006 |  | 706 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_8000-20000 | 2770 |  | 706 |  |

### `few_shot_target`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| adaptation_days | `stratia-1.0.0-demo` | n | 273 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | etage_set_agreement | 0.031 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | c2_etage_ratio | 0.034 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | oktas_within1 | 0.278 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | c2_oktas_ratio | 0.361 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | oktas_mae | 2.608 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | oktas_bias | 2.396 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | genus_top1_in_set | 0.300 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | etage_f1 | 0.514 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | cloud_fraction_corr_vs_cover | 0.833 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | cloud_fraction_mae_vs_cover | 0.107 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | cloud_fraction_on_le1_okta | 0.360 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | glare_fraction_strong_frames | 0.000 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | ood_flagged_share | 0.084 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | ood_mean | 1.127 |  | 273 |  |
| adaptation_days | `stratia-1.0.0-demo` | etage_set_agreement_majority | 0.022 | [0.005, 0.040] | 273 | frame bootstrap, one day |
| adaptation_days | `stratia-1.0.0-demo` | oktas_within1_majority | 0.245 | [0.194, 0.297] | 273 | frame bootstrap, one day |
| adaptation_days | `stratia-1.0.1-demo` | n | 273 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | etage_set_agreement | 0.905 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | c2_etage_ratio | 0.993 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | oktas_within1 | 0.681 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | c2_oktas_ratio | 0.884 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | oktas_mae | 1.187 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | oktas_bias | 0.319 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | genus_top1_in_set | 0.989 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | etage_f1 | 0.971 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | cloud_fraction_corr_vs_cover | 0.833 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | cloud_fraction_mae_vs_cover | 0.107 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | cloud_fraction_on_le1_okta | 0.360 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | glare_fraction_strong_frames | 0.000 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | ood_flagged_share | 0.015 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | ood_mean | 0.798 |  | 273 |  |
| adaptation_days | `stratia-1.0.1-demo` | etage_set_agreement_majority | 0.901 | [0.868, 0.934] | 273 | frame bootstrap, one day |
| adaptation_days | `stratia-1.0.1-demo` | oktas_within1_majority | 0.718 | [0.663, 0.775] | 273 | frame bootstrap, one day |

Not evaluated (no model exists): `lodo/swim/test`, `held_out_station/same_days/test`, `held_out_station/disjoint_days/test`.
