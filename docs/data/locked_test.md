# Locked-test evaluation (P090, the one run)

Split `test`; lock `c41841f43cc488d7`; run `20261025T113929Z_locked_test_test_f2f212f294`; git `7d4b7a4`; 500 cluster-bootstrap resamples; models and protocol as pre-registered in `docs/phases/P090-locked-test.md`. Thresholds 0.5; one seed except the in-domain global heads (three).

## The Gate G4 criteria on these rows

| Criterion | Quantity | P082 (val) | This split | 95 % CI | Reference | Required | Holds |
|---|---|---|---|---|---|---|---|
| C1 | LODO étage macro-F1, mean of the four folds | 0.548 | 0.540 |  | 0.529 (probe ViT-B/16 @ 224, refitted) | ≥ reference + 0.05 | no |
| C1 | in-domain genus macro-F1 (CCSN, MGCD) | 0.719 | 0.706 |  | 0.733 (probe ViT-L/16 @ 224, refitted) | ≥ reference − 0.01 | no |
| C2 | Montenegro étage-set agreement / human ceiling | 1.047 | 0.926 | [0.923, 0.928] | 0.900 (pre-registered bar) | ≥ 0.90 | yes |
| C2 | Montenegro oktas ±1 agreement / human ceiling | 1.027 | 0.963 | [0.958, 0.966] | 0.900 (pre-registered bar) | ≥ 0.90 | yes |
| C3 | CBH skill, OLDLR (CDLRA) pairs | 0.471 | 0.536 | [0.440, 0.630] | 0.200 (pre-registered bar) | ≥ 0.20 | yes |
| C3 | CBH skill, WESTE (CDLRB) pairs, OLDLR-only training | 0.448 | 0.436 | [0.276, 0.585] | 0.100 (pre-registered bar) | ≥ 0.10 | yes |
| C3 | CBH MAE (m), all pairs, against the frozen-feature ridge | 566 | 872 | [679, 1055] | 1096 (ridge DINOv3 ViT-S/16 @ 512, refitted (the P058 CNN kept no checkpoint; 724 m on val)) | ≤ 0.9 × reference | yes |
| C3 | 80 % interval coverage with the P089 temperature and radius | 0.802 | 0.720 |  | 0.800 (nominal) | 0.75–0.85 | no |

## Every number

### `in_domain`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| almeria | `indomain_e2s_s0` | etage_f1 | 0.803 | [0.678, 0.862] | 164 |  |
| almeria | `indomain_e2s_s0` | etage_unsupported_fp | - |  | 164 |  |
| almeria | `indomain_e2s_s0` | etage_f1_tuned | 0.815 | [0.726, 0.886] | 164 |  |
| almeria | `indomain_e2s_s0` | n_etage | 164 | [116, 216] | 164 |  |
| ccsn | `indomain_e2s_s0` | etage_f1 | 0.719 | [0.676, 0.756] | 512 |  |
| ccsn | `indomain_e2s_s0` | etage_unsupported_fp | - |  | 512 |  |
| ccsn | `indomain_e2s_s0` | etage_f1_tuned | 0.746 | [0.718, 0.785] | 512 |  |
| ccsn | `indomain_e2s_s0` | n_etage | 512 | [502, 523] | 512 |  |
| ccsn | `indomain_e2s_s0` | genus_f1 | 0.591 | [0.544, 0.629] | 512 |  |
| ccsn | `indomain_e2s_s0` | genus_acc | 0.600 | [0.556, 0.643] | 512 |  |
| ccsn | `indomain_e2s_s0` | n_genus | 512 | [502, 523] | 512 |  |
| mgcd | `indomain_e2s_s0` | etage_f1 | 1.000 | [1.000, 1.000] | 1569 |  |
| mgcd | `indomain_e2s_s0` | etage_unsupported_fp | 0.000 | [0.000, 0.000] | 1569 |  |
| mgcd | `indomain_e2s_s0` | etage_f1_tuned | 1.000 | [1.000, 1.000] | 1569 |  |
| mgcd | `indomain_e2s_s0` | n_etage | 816 | [291, 1380] | 1569 |  |
| mgcd | `indomain_e2s_s0` | genus_f1 | 0.821 | [0.706, 0.997] | 1569 |  |
| mgcd | `indomain_e2s_s0` | genus_acc | 0.879 | [0.648, 0.999] | 1569 |  |
| mgcd | `indomain_e2s_s0` | n_genus | 1380 | [761, 2069] | 1569 |  |
| montenegro | `indomain_e2s_s0` | etage_f1 | 0.680 | [0.629, 0.725] | 446 |  |
| montenegro | `indomain_e2s_s0` | etage_unsupported_fp | - |  | 446 |  |
| montenegro | `indomain_e2s_s0` | etage_f1_tuned | 0.700 | [0.696, 0.749] | 446 |  |
| montenegro | `indomain_e2s_s0` | n_etage | 446 | [318, 574] | 446 |  |
| montenegro | `indomain_e2s_s0` | oktas_mae | 0.706 | [0.648, 0.811] | 446 |  |
| montenegro | `indomain_e2s_s0` | oktas_within1 | 0.841 | [0.836, 0.843] | 446 |  |
| montenegro | `indomain_e2s_s0` | oktas_kl | 0.662 | [0.610, 0.757] | 446 |  |
| montenegro | `indomain_e2s_s0` | etage_set_agreement | 0.789 | [0.786, 0.791] | 446 |  |
| montenegro | `indomain_e2s_s0` | c2_etage_ratio | 0.926 | [0.923, 0.928] | 446 |  |
| montenegro | `indomain_e2s_s0` | c2_oktas_ratio | 0.963 | [0.958, 0.966] | 446 |  |
| almeria | `indomain_e2s_s1` | etage_f1 | 0.801 |  | 164 |  |
| almeria | `indomain_e2s_s1` | etage_unsupported_fp | - |  | 164 |  |
| almeria | `indomain_e2s_s1` | etage_f1_tuned | 0.810 |  | 164 |  |
| almeria | `indomain_e2s_s1` | n_etage | 164 |  | 164 |  |
| ccsn | `indomain_e2s_s1` | etage_f1 | 0.732 |  | 512 |  |
| ccsn | `indomain_e2s_s1` | etage_unsupported_fp | - |  | 512 |  |
| ccsn | `indomain_e2s_s1` | etage_f1_tuned | 0.755 |  | 512 |  |
| ccsn | `indomain_e2s_s1` | n_etage | 512 |  | 512 |  |
| ccsn | `indomain_e2s_s1` | genus_f1 | 0.598 |  | 512 |  |
| ccsn | `indomain_e2s_s1` | genus_acc | 0.611 |  | 512 |  |
| ccsn | `indomain_e2s_s1` | n_genus | 512 |  | 512 |  |
| mgcd | `indomain_e2s_s1` | etage_f1 | 1.000 |  | 1569 |  |
| mgcd | `indomain_e2s_s1` | etage_unsupported_fp | 0.000 |  | 1569 |  |
| mgcd | `indomain_e2s_s1` | etage_f1_tuned | 1.000 |  | 1569 |  |
| mgcd | `indomain_e2s_s1` | n_etage | 816 |  | 1569 |  |
| mgcd | `indomain_e2s_s1` | genus_f1 | 0.838 |  | 1569 |  |
| mgcd | `indomain_e2s_s1` | genus_acc | 0.887 |  | 1569 |  |
| mgcd | `indomain_e2s_s1` | n_genus | 1380 |  | 1569 |  |
| montenegro | `indomain_e2s_s1` | etage_f1 | 0.664 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | etage_unsupported_fp | - |  | 446 |  |
| montenegro | `indomain_e2s_s1` | etage_f1_tuned | 0.694 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | n_etage | 446 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | oktas_mae | 0.724 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | oktas_within1 | 0.839 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | oktas_kl | 0.657 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | etage_set_agreement | 0.783 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | c2_etage_ratio | 0.918 |  | 446 |  |
| montenegro | `indomain_e2s_s1` | c2_oktas_ratio | 0.961 |  | 446 |  |
| almeria | `indomain_e2s_s2` | etage_f1 | 0.754 |  | 164 |  |
| almeria | `indomain_e2s_s2` | etage_unsupported_fp | - |  | 164 |  |
| almeria | `indomain_e2s_s2` | etage_f1_tuned | 0.797 |  | 164 |  |
| almeria | `indomain_e2s_s2` | n_etage | 164 |  | 164 |  |
| ccsn | `indomain_e2s_s2` | etage_f1 | 0.738 |  | 512 |  |
| ccsn | `indomain_e2s_s2` | etage_unsupported_fp | - |  | 512 |  |
| ccsn | `indomain_e2s_s2` | etage_f1_tuned | 0.749 |  | 512 |  |
| ccsn | `indomain_e2s_s2` | n_etage | 512 |  | 512 |  |
| ccsn | `indomain_e2s_s2` | genus_f1 | 0.593 |  | 512 |  |
| ccsn | `indomain_e2s_s2` | genus_acc | 0.602 |  | 512 |  |
| ccsn | `indomain_e2s_s2` | n_genus | 512 |  | 512 |  |
| mgcd | `indomain_e2s_s2` | etage_f1 | 0.999 |  | 1569 |  |
| mgcd | `indomain_e2s_s2` | etage_unsupported_fp | 0.000 |  | 1569 |  |
| mgcd | `indomain_e2s_s2` | etage_f1_tuned | 1.000 |  | 1569 |  |
| mgcd | `indomain_e2s_s2` | n_etage | 816 |  | 1569 |  |
| mgcd | `indomain_e2s_s2` | genus_f1 | 0.826 |  | 1569 |  |
| mgcd | `indomain_e2s_s2` | genus_acc | 0.880 |  | 1569 |  |
| mgcd | `indomain_e2s_s2` | n_genus | 1380 |  | 1569 |  |
| montenegro | `indomain_e2s_s2` | etage_f1 | 0.665 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | etage_unsupported_fp | - |  | 446 |  |
| montenegro | `indomain_e2s_s2` | etage_f1_tuned | 0.704 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | n_etage | 446 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | oktas_mae | 0.697 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | oktas_within1 | 0.843 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | oktas_kl | 0.649 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | etage_set_agreement | 0.780 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | c2_etage_ratio | 0.916 |  | 446 |  |
| montenegro | `indomain_e2s_s2` | c2_oktas_ratio | 0.966 |  | 446 |  |
| almeria | `indomain_e2s_mean3` | etage_f1 | 0.786 |  |  | std over 3 seeds 0.0226 |
| almeria | `indomain_e2s_mean3` | etage_f1_tuned | 0.807 |  |  | std over 3 seeds 0.0075 |
| almeria | `indomain_e2s_mean3` | n_etage | 164 |  |  | std over 3 seeds 0.0000 |
| ccsn | `indomain_e2s_mean3` | etage_f1 | 0.730 |  |  | std over 3 seeds 0.0080 |
| ccsn | `indomain_e2s_mean3` | etage_f1_tuned | 0.750 |  |  | std over 3 seeds 0.0036 |
| ccsn | `indomain_e2s_mean3` | genus_acc | 0.604 |  |  | std over 3 seeds 0.0051 |
| ccsn | `indomain_e2s_mean3` | genus_f1 | 0.594 |  |  | std over 3 seeds 0.0029 |
| ccsn | `indomain_e2s_mean3` | n_etage | 512 |  |  | std over 3 seeds 0.0000 |
| ccsn | `indomain_e2s_mean3` | n_genus | 512 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | etage_f1 | 1.000 |  |  | std over 3 seeds 0.0005 |
| mgcd | `indomain_e2s_mean3` | etage_f1_tuned | 1.000 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | etage_unsupported_fp | 0.000 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | genus_acc | 0.882 |  |  | std over 3 seeds 0.0036 |
| mgcd | `indomain_e2s_mean3` | genus_f1 | 0.828 |  |  | std over 3 seeds 0.0070 |
| mgcd | `indomain_e2s_mean3` | n_etage | 816 |  |  | std over 3 seeds 0.0000 |
| mgcd | `indomain_e2s_mean3` | n_genus | 1380 |  |  | std over 3 seeds 0.0000 |
| montenegro | `indomain_e2s_mean3` | c2_etage_ratio | 0.920 |  |  | std over 3 seeds 0.0045 |
| montenegro | `indomain_e2s_mean3` | c2_oktas_ratio | 0.963 |  |  | std over 3 seeds 0.0021 |
| montenegro | `indomain_e2s_mean3` | etage_f1 | 0.670 |  |  | std over 3 seeds 0.0076 |
| montenegro | `indomain_e2s_mean3` | etage_f1_tuned | 0.699 |  |  | std over 3 seeds 0.0040 |
| montenegro | `indomain_e2s_mean3` | etage_set_agreement | 0.784 |  |  | std over 3 seeds 0.0038 |
| montenegro | `indomain_e2s_mean3` | n_etage | 446 |  |  | std over 3 seeds 0.0000 |
| montenegro | `indomain_e2s_mean3` | oktas_kl | 0.656 |  |  | std over 3 seeds 0.0054 |
| montenegro | `indomain_e2s_mean3` | oktas_mae | 0.709 |  |  | std over 3 seeds 0.0112 |
| montenegro | `indomain_e2s_mean3` | oktas_within1 | 0.841 |  |  | std over 3 seeds 0.0018 |
| ccsn | `probe_dinov3_vitl16_224` | genus_f1 | 0.549 | [0.501, 0.591] | 512 | C 0.01 chosen on val |
| ccsn | `probe_dinov3_vitl16_224` | genus_acc | 0.568 |  | 512 |  |
| mgcd | `probe_dinov3_vitl16_224` | genus_f1 | 0.918 | [0.763, 0.949] | 1569 | C 0.01 chosen on val |
| mgcd | `probe_dinov3_vitl16_224` | genus_acc | 0.930 |  | 1569 |  |

### `cbh_pairs_etage`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| eye2sky_cbh | `indomain_e2s_s0` | etage_recall | 0.964 | [0.946, 0.984] | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s0` | lowest_etage_acc | 0.704 | [0.618, 0.778] | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s0` | n_etage | 2672 | [2258, 3144] | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s1` | etage_recall | 0.964 |  | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s1` | lowest_etage_acc | 0.713 |  | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s1` | n_etage | 2672 |  | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s2` | etage_recall | 0.962 |  | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s2` | lowest_etage_acc | 0.723 |  | 2672 |  |
| eye2sky_cbh | `indomain_e2s_s2` | n_etage | 2672 |  | 2672 |  |
| eye2sky_cbh | `indomain_e2s_mean3` | etage_recall | 0.963 |  |  | std over 3 seeds 0.0012 |
| eye2sky_cbh | `indomain_e2s_mean3` | lowest_etage_acc | 0.714 |  |  | std over 3 seeds 0.0079 |
| eye2sky_cbh | `indomain_e2s_mean3` | n_etage | 2672 |  |  | std over 3 seeds 0.0000 |

### `ood`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| in_domain_all | `indomain_e2s_s0` | flagged_share | 0.054 |  | 5520 |  |
| almeria | `indomain_e2s_s0` | flagged_share | 0.055 |  | 164 |  |
| ccsn | `indomain_e2s_s0` | flagged_share | 0.430 |  | 512 |  |
| eye2sky | `indomain_e2s_s0` | flagged_share | 0.000 |  | 2672 |  |
| mgcd | `indomain_e2s_s0` | flagged_share | 0.022 |  | 1569 |  |
| montenegro | `indomain_e2s_s0` | flagged_share | 0.004 |  | 446 |  |
| swim | `indomain_e2s_s0` | flagged_share | 0.191 |  | 157 |  |
| b0268 | `indomain_e2s_s0` | flagged_share | 0.400 |  | 120 |  |
| b0268_vs_in_domain | `indomain_e2s_s0` | auroc | 0.939 |  | 5640 | higher score = out of domain; cached-feature path for both |

### `lodo`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| ccsn | `lodo_ccsn_e2s` | etage_f1 | 0.451 | [0.430, 0.471] | 2288 |  |
| ccsn | `lodo_ccsn_e2s` | etage_unsupported_fp | - |  | 2288 |  |
| ccsn | `lodo_ccsn_e2s` | etage_f1_tuned | 0.534 | [0.518, 0.553] | 2288 |  |
| ccsn | `lodo_ccsn_e2s` | n_etage | 2109 | [2086, 2132] | 2288 |  |
| ccsn | `lodo_ccsn_e2s` | genus_f1 | 0.091 | [0.081, 0.100] | 2288 |  |
| ccsn | `lodo_ccsn_e2s` | genus_acc | 0.156 | [0.140, 0.171] | 2288 |  |
| ccsn | `lodo_ccsn_e2s` | n_genus | 2288 | [2263, 2312] | 2288 |  |
| ccsn | `lodo_ccsn_e2s` | etage_f1_valtuned | 0.533 |  | 2109 | thresholds chosen on the fold's val rows |
| mgcd | `lodo_mgcd_e2s` | etage_f1 | 0.709 | [0.620, 0.776] | 7041 |  |
| mgcd | `lodo_mgcd_e2s` | etage_unsupported_fp | 0.056 | [0.025, 0.097] | 7041 |  |
| mgcd | `lodo_mgcd_e2s` | etage_f1_tuned | 0.822 | [0.752, 0.875] | 7041 |  |
| mgcd | `lodo_mgcd_e2s` | n_etage | 3661 | [2638, 4765] | 7041 |  |
| mgcd | `lodo_mgcd_e2s` | genus_f1 | 0.517 | [0.443, 0.578] | 7041 |  |
| mgcd | `lodo_mgcd_e2s` | genus_acc | 0.540 | [0.447, 0.627] | 7041 |  |
| mgcd | `lodo_mgcd_e2s` | n_genus | 6191 | [4769, 7544] | 7041 |  |
| mgcd | `lodo_mgcd_e2s` | etage_f1_valtuned | 0.822 |  | 3661 | thresholds chosen on the fold's val rows |
| montenegro | `lodo_montenegro_e2s` | etage_f1 | 0.367 | [0.298, 0.446] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | etage_unsupported_fp | - |  | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | etage_f1_tuned | 0.492 | [0.426, 0.544] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | n_etage | 2011 | [1720, 2288] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | oktas_mae | 4.248 | [3.774, 4.829] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | oktas_within1 | 0.311 | [0.249, 0.362] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | oktas_kl | 1.645 | [1.575, 1.738] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | etage_set_agreement | 0.293 | [0.219, 0.381] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | c2_etage_ratio | 0.344 | [0.257, 0.448] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | c2_oktas_ratio | 0.357 | [0.285, 0.415] | 2011 |  |
| montenegro | `lodo_montenegro_e2s` | etage_f1_valtuned | 0.473 |  | 2011 | thresholds chosen on the fold's val rows |
| almeria | `lodo_almeria_e2s` | etage_f1 | 0.634 | [0.599, 0.660] | 734 |  |
| almeria | `lodo_almeria_e2s` | etage_unsupported_fp | - |  | 734 |  |
| almeria | `lodo_almeria_e2s` | etage_f1_tuned | 0.646 | [0.620, 0.681] | 734 |  |
| almeria | `lodo_almeria_e2s` | n_etage | 734 | [644, 832] | 734 |  |
| almeria | `lodo_almeria_e2s` | etage_f1_valtuned | 0.633 |  | 734 | thresholds chosen on the fold's val rows |
| ccsn | `probe_dinov3_vitb16_224` | etage_f1 | 0.396 | [0.376, 0.414] | 2109 | C 0.1 chosen on val |
| ccsn | `probe_dinov3_vitb16_224` | etage_unsupported_fp | - |  | 2109 |  |
| mgcd | `probe_dinov3_vitb16_224` | etage_f1 | 0.860 | [0.793, 0.917] | 3661 | C 0.01 chosen on val |
| mgcd | `probe_dinov3_vitb16_224` | etage_unsupported_fp | 0.009 |  | 3661 |  |
| montenegro | `probe_dinov3_vitb16_224` | etage_f1 | 0.463 | [0.392, 0.525] | 2011 | C 0.01 chosen on val |
| montenegro | `probe_dinov3_vitb16_224` | etage_unsupported_fp | - |  | 2011 |  |
| almeria | `probe_dinov3_vitb16_224` | etage_f1 | 0.397 | [0.361, 0.425] | 734 | C 0.1 chosen on val |
| almeria | `probe_dinov3_vitb16_224` | etage_unsupported_fp | - |  | 734 |  |

### `dense`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| all | `dense_sky_layer_s0` | sky_miou | 0.900 | [0.897, 0.903] | 1775 |  |
| all | `dense_sky_layer_s0` | iou_invalid | 0.964 |  | 1775 |  |
| all | `dense_sky_layer_s0` | iou_sky | 0.859 |  | 1775 |  |
| all | `dense_sky_layer_s0` | iou_cloud | 0.878 |  | 1775 |  |
| all | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.528 |  | 1775 |  |
| all | `dense_sky_layer_s0` | layer_miou | 0.668 | [0.596, 0.743] | 1775 |  |
| all | `dense_sky_layer_s0` | layer_iou_low | 0.783 |  | 1775 |  |
| all | `dense_sky_layer_s0` | layer_iou_mid | 0.454 |  | 1775 |  |
| all | `dense_sky_layer_s0` | layer_iou_high | 0.766 |  | 1775 |  |
| almeria | `dense_sky_layer_s0` | sky_miou | 0.885 | [0.871, 0.896] | 164 |  |
| almeria | `dense_sky_layer_s0` | iou_invalid | 0.964 |  | 164 |  |
| almeria | `dense_sky_layer_s0` | iou_sky | 0.867 |  | 164 |  |
| almeria | `dense_sky_layer_s0` | iou_cloud | 0.824 |  | 164 |  |
| almeria | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.335 |  | 164 |  |
| almeria | `dense_sky_layer_s0` | layer_miou | 0.668 | [0.597, 0.739] | 164 |  |
| almeria | `dense_sky_layer_s0` | layer_iou_low | 0.783 |  | 164 |  |
| almeria | `dense_sky_layer_s0` | layer_iou_mid | 0.454 |  | 164 |  |
| almeria | `dense_sky_layer_s0` | layer_iou_high | 0.766 |  | 164 |  |
| shwimseg | `dense_sky_layer_s0` | sky_miou | 0.822 | [0.776, 0.860] | 31 |  |
| shwimseg | `dense_sky_layer_s0` | iou_sky | 0.806 |  | 31 |  |
| shwimseg | `dense_sky_layer_s0` | iou_cloud | 0.839 |  | 31 |  |
| shwimseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.553 |  | 31 |  |
| swimseg | `dense_sky_layer_s0` | sky_miou | 0.865 | [0.848, 0.879] | 203 |  |
| swimseg | `dense_sky_layer_s0` | iou_sky | 0.848 |  | 203 |  |
| swimseg | `dense_sky_layer_s0` | iou_cloud | 0.882 |  | 203 |  |
| swimseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.568 |  | 203 |  |
| swinseg | `dense_sky_layer_s0` | sky_miou | 0.866 | [0.840, 0.892] | 23 |  |
| swinseg | `dense_sky_layer_s0` | iou_sky | 0.867 |  | 23 |  |
| swinseg | `dense_sky_layer_s0` | iou_cloud | 0.865 |  | 23 |  |
| swinseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.497 |  | 23 |  |
| swinyseg | `dense_sky_layer_s0` | sky_miou | 0.872 | [0.866, 0.877] | 1354 |  |
| swinyseg | `dense_sky_layer_s0` | iou_sky | 0.861 |  | 1354 |  |
| swinyseg | `dense_sky_layer_s0` | iou_cloud | 0.883 |  | 1354 |  |
| swinyseg | `dense_sky_layer_s0` | predicted_cloud_fraction | 0.546 |  | 1354 |  |
| clear_sky_test | `dense_sky_layer_s0` | fraction_invalid | 0.000 |  | 200 |  |
| clear_sky_test | `dense_sky_layer_s0` | fraction_sky | 0.719 |  | 200 |  |
| clear_sky_test | `dense_sky_layer_s0` | fraction_cloud | 0.281 |  | 200 |  |
| clear_sky_test | `dense_sky_layer_s0` | fraction_sun_glare | 0.000 |  | 200 |  |

### `cbh_pairs`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| all | `climatology_train` | mae_m | 2012 |  | 2965 | clim 1726 m |
| all | `cbh_base_s0` | n | 2965 |  | 4383 |  |
| all | `cbh_base_s0` | mae_m | 872 | [679, 1055] | 4383 |  |
| all | `cbh_base_s0` | bias_m | -106 |  | 4383 |  |
| all | `cbh_base_s0` | rmsd_m | 1524 |  | 4383 |  |
| all | `cbh_base_s0` | skill | 0.566 | [0.476, 0.663] | 4383 |  |
| all | `cbh_base_s0` | within_250m | 0.363 |  | 4383 |  |
| all | `cbh_base_s0` | within_500m | 0.546 | [0.456, 0.644] | 4383 |  |
| all | `cbh_base_s0` | within_1000m | 0.747 |  | 4383 |  |
| all | `cbh_base_s0` | mae_0-1000 | 339 |  | 4383 |  |
| all | `cbh_base_s0` | mae_1000-2000 | 410 |  | 4383 |  |
| all | `cbh_base_s0` | mae_2000-4000 | 853 |  | 4383 |  |
| all | `cbh_base_s0` | mae_4000-8000 | 1785 |  | 4383 |  |
| all | `cbh_base_s0` | mae_8000-20000 | 1923 |  | 4383 |  |
| all | `cbh_base_s0` | crps | 0.035 | [0.029, 0.041] | 4383 |  |
| all | `cbh_base_s0` | coverage_80 | 0.878 | [0.847, 0.904] | 4383 |  |
| all | `cbh_base_s0` | cloud_balanced_acc | 0.915 |  | 4383 |  |
| all | `cbh_base_s0` | top_bin_share_pred | 0.210 |  | 4383 |  |
| all | `cbh_base_s0` | top_bin_share_true | 0.170 |  | 4383 |  |
| all | `cbh_base_s0` | entropy_ratio | 0.903 |  | 4383 |  |
| all | `cbh_base_s0` | n_cloudy | 2965 |  | 4383 |  |
| all | `cbh_base_s0` | coverage_80_calibrated | 0.720 |  | 2965 | P089 temperature 1.2394 and radius 0.3265 |
| all | `cbh_base_s0` | coverage_95_calibrated | 0.918 |  | 2965 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0` | n | 1466 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_m | 933 | [744, 1127] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | bias_m | -46.408 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | rmsd_m | 1621 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | skill | 0.536 | [0.440, 0.630] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | within_250m | 0.338 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | within_500m | 0.531 | [0.448, 0.618] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | within_1000m | 0.734 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_0-1000 | 432 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_1000-2000 | 449 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_2000-4000 | 838 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_4000-8000 | 1935 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | mae_8000-20000 | 1837 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | crps | 0.038 | [0.032, 0.046] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | coverage_80 | 0.868 | [0.832, 0.900] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | cloud_balanced_acc | 0.913 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | top_bin_share_pred | 0.236 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | top_bin_share_true | 0.158 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | entropy_ratio | 0.893 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | n_cloudy | 1466 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0` | coverage_80_calibrated | 0.698 |  | 1466 | P089 temperature 1.2394 and radius 0.3265 |
| OLDLR | `cbh_base_s0` | coverage_95_calibrated | 0.911 |  | 1466 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0` | skill_station_climatology | 0.539 |  | 2143 | secondary: against the climatology's MAE on this station's rows (2024 m) |
| WESTE | `cbh_base_s0` | n | 1499 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_m | 813 | [601, 1013] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | bias_m | -164 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | rmsd_m | 1424 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | skill | 0.596 | [0.496, 0.701] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | within_250m | 0.386 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | within_500m | 0.561 | [0.458, 0.673] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | within_1000m | 0.760 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_0-1000 | 249 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_1000-2000 | 376 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_2000-4000 | 869 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_4000-8000 | 1626 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | mae_8000-20000 | 2004 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | crps | 0.032 | [0.026, 0.038] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | coverage_80 | 0.887 | [0.855, 0.920] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | cloud_balanced_acc | 0.916 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | top_bin_share_pred | 0.239 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | top_bin_share_true | 0.183 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | entropy_ratio | 0.905 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | n_cloudy | 1499 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0` | coverage_80_calibrated | 0.742 |  | 1499 | P089 temperature 1.2394 and radius 0.3265 |
| WESTE | `cbh_base_s0` | coverage_95_calibrated | 0.925 |  | 1499 | P089 temperature 1.2394 and radius 0.4376 |
| WESTE | `cbh_base_s0` | skill_station_climatology | 0.593 |  | 2240 | secondary: against the climatology's MAE on this station's rows (1999 m) |
| all | `cbh_base_s0_geometry_zeroed` | n | 2965 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_m | 875 | [680, 1050] | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | bias_m | -11.869 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | rmsd_m | 1526 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | skill | 0.565 | [0.478, 0.662] | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | within_250m | 0.354 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | within_500m | 0.546 | [0.455, 0.651] | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | within_1000m | 0.741 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_0-1000 | 397 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_1000-2000 | 423 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_2000-4000 | 896 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_4000-8000 | 1740 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | mae_8000-20000 | 1774 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | crps | 0.035 | [0.029, 0.040] | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | coverage_80 | 0.905 | [0.873, 0.928] | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | cloud_balanced_acc | 0.910 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | top_bin_share_pred | 0.234 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | top_bin_share_true | 0.170 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | entropy_ratio | 0.863 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | n_cloudy | 2965 |  | 4383 |  |
| all | `cbh_base_s0_geometry_zeroed` | coverage_80_calibrated | 0.764 |  | 2965 | P089 temperature 1.2394 and radius 0.3265 |
| all | `cbh_base_s0_geometry_zeroed` | coverage_95_calibrated | 0.943 |  | 2965 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0_geometry_zeroed` | n | 1466 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_m | 941 | [746, 1129] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | bias_m | 51.963 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | rmsd_m | 1637 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | skill | 0.532 | [0.439, 0.629] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | within_250m | 0.327 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | within_500m | 0.529 | [0.441, 0.625] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | within_1000m | 0.729 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_0-1000 | 528 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_1000-2000 | 458 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_2000-4000 | 881 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_4000-8000 | 1866 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | mae_8000-20000 | 1720 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | crps | 0.038 | [0.032, 0.044] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | coverage_80 | 0.901 | [0.867, 0.928] | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | cloud_balanced_acc | 0.910 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | top_bin_share_pred | 0.254 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | top_bin_share_true | 0.158 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | entropy_ratio | 0.855 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | n_cloudy | 1466 |  | 2143 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| OLDLR | `cbh_base_s0_geometry_zeroed` | coverage_80_calibrated | 0.744 |  | 1466 | P089 temperature 1.2394 and radius 0.3265 |
| OLDLR | `cbh_base_s0_geometry_zeroed` | coverage_95_calibrated | 0.935 |  | 1466 | P089 temperature 1.2394 and radius 0.4376 |
| OLDLR | `cbh_base_s0_geometry_zeroed` | skill_station_climatology | 0.535 |  | 2143 | secondary: against the climatology's MAE on this station's rows (2024 m) |
| WESTE | `cbh_base_s0_geometry_zeroed` | n | 1499 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_m | 811 | [606, 1004] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | bias_m | -74.296 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | rmsd_m | 1408 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | skill | 0.597 | [0.501, 0.699] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | within_250m | 0.382 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | within_500m | 0.564 | [0.461, 0.673] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | within_1000m | 0.754 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_0-1000 | 272 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_1000-2000 | 391 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_2000-4000 | 912 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_4000-8000 | 1606 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | mae_8000-20000 | 1825 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | crps | 0.032 | [0.026, 0.037] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | coverage_80 | 0.909 | [0.881, 0.937] | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | cloud_balanced_acc | 0.911 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | top_bin_share_pred | 0.268 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | top_bin_share_true | 0.183 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | entropy_ratio | 0.862 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | n_cloudy | 1499 |  | 2240 | skill against the training climatology's MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_base_s0_geometry_zeroed` | coverage_80_calibrated | 0.783 |  | 1499 | P089 temperature 1.2394 and radius 0.3265 |
| WESTE | `cbh_base_s0_geometry_zeroed` | coverage_95_calibrated | 0.951 |  | 1499 | P089 temperature 1.2394 and radius 0.4376 |
| WESTE | `cbh_base_s0_geometry_zeroed` | skill_station_climatology | 0.595 |  | 2240 | secondary: against the climatology's MAE on this station's rows (1999 m) |
| WESTE | `cbh_oldlr_only_s0` | n | 1499 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_m | 1134 | [835, 1456] | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | bias_m | -575 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | rmsd_m | 1889 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | skill | 0.436 | [0.276, 0.585] | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | within_250m | 0.276 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | within_500m | 0.496 | [0.403, 0.592] | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | within_1000m | 0.696 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_0-1000 | 364 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_1000-2000 | 501 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_2000-4000 | 868 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_4000-8000 | 1875 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | mae_8000-20000 | 4214 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | crps | 0.049 | [0.040, 0.058] | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | coverage_80 | 0.799 | [0.754, 0.847] | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | cloud_balanced_acc | 0.731 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | top_bin_share_pred | 0.240 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | top_bin_share_true | 0.183 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | entropy_ratio | 0.819 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | n_cloudy | 1499 |  | 2240 | site never seen in training; OLDLR training climatology (1750 m), its MAE on the whole split (Gate G4 definition) |
| WESTE | `cbh_oldlr_only_s0` | coverage_80_calibrated | 0.605 |  | 1499 | P089 temperature 1.2394 and radius 0.3265 |
| WESTE | `cbh_oldlr_only_s0` | coverage_95_calibrated | 0.855 |  | 1499 | P089 temperature 1.2394 and radius 0.4376 |
| WESTE | `cbh_oldlr_only_s0` | skill_station_climatology | 0.433 |  | 2240 | secondary: against the OLDLR climatology's MAE on the WESTE rows (1999 m) |
| all | `ridge_dinov3_vits16_512` | n | 2965 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_m | 1096 | [852, 1365] | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | bias_m | -285 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | rmsd_m | 1816 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | skill | 0.455 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | within_250m | 0.295 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | within_500m | 0.470 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | within_1000m | 0.667 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_0-1000 | 428 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_1000-2000 | 472 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_2000-4000 | 909 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_4000-8000 | 2127 |  | 2965 | alpha 1000.0 chosen on val |
| all | `ridge_dinov3_vits16_512` | mae_8000-20000 | 3122 |  | 2965 | alpha 1000.0 chosen on val |
| OLDLR | `ridge_dinov3_vits16_512` | n | 1466 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_m | 1117 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | bias_m | -212 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | rmsd_m | 1859 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | skill | 0.448 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | within_250m | 0.286 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | within_500m | 0.471 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | within_1000m | 0.669 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_0-1000 | 467 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_1000-2000 | 491 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_2000-4000 | 894 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_4000-8000 | 2079 |  | 1466 |  |
| OLDLR | `ridge_dinov3_vits16_512` | mae_8000-20000 | 3187 |  | 1466 |  |
| WESTE | `ridge_dinov3_vits16_512` | n | 1499 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_m | 1075 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | bias_m | -356 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | rmsd_m | 1773 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | skill | 0.462 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | within_250m | 0.304 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | within_500m | 0.469 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | within_1000m | 0.664 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_0-1000 | 391 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_1000-2000 | 456 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_2000-4000 | 926 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_4000-8000 | 2177 |  | 1499 |  |
| WESTE | `ridge_dinov3_vits16_512` | mae_8000-20000 | 3060 |  | 1499 |  |

### `few_shot_target`

| Set | Model | Metric | Value | 95 % CI | n | Note |
|---|---|---|---|---|---|---|
| locked_day | `stratia-1.0.0-demo` | n | 120 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | etage_set_agreement | 0.000 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | c2_etage_ratio | 0.000 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | oktas_within1 | 0.017 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | c2_oktas_ratio | 0.022 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | oktas_mae | 3.458 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | oktas_bias | 3.458 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | genus_top1_in_set | 0.121 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | etage_f1 | 0.217 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | cloud_fraction_corr_vs_cover | 0.776 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | cloud_fraction_mae_vs_cover | 0.079 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | glare_fraction_strong_frames | 0.000 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | ood_flagged_share | 0.508 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | ood_mean | 1.451 |  | 120 |  |
| locked_day | `stratia-1.0.0-demo` | etage_set_agreement_majority | 0.000 | [0.000, 0.000] | 120 | frame bootstrap, one day |
| locked_day | `stratia-1.0.0-demo` | oktas_within1_majority | 0.000 | [0.000, 0.000] | 120 | frame bootstrap, one day |
| locked_day | `stratia-1.0.1-demo` | n | 120 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | etage_set_agreement | 0.463 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | c2_etage_ratio | 0.508 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | oktas_within1 | 0.671 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | c2_oktas_ratio | 0.870 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | oktas_mae | 1.000 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | oktas_bias | -0.500 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | genus_top1_in_set | 1.000 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | etage_f1 | 0.821 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | cloud_fraction_corr_vs_cover | 0.776 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | cloud_fraction_mae_vs_cover | 0.079 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | glare_fraction_strong_frames | 0.000 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | ood_flagged_share | 0.000 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | ood_mean | 0.832 |  | 120 |  |
| locked_day | `stratia-1.0.1-demo` | etage_set_agreement_majority | 0.458 | [0.375, 0.546] | 120 | frame bootstrap, one day |
| locked_day | `stratia-1.0.1-demo` | oktas_within1_majority | 0.900 | [0.842, 0.950] | 120 | frame bootstrap, one day |

Not evaluated (no model exists): `lodo/swim/test`, `held_out_station/same_days/test`, `held_out_station/disjoint_days/test`.
