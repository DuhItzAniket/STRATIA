# B0268 few-shot adaptation (P084, reduced)

Heads-only LP-FT on cached DINOv3 ViT-S/16 features from the Gate G4 global run `20261011T125754Z_global_indomain_e2s_s0_27d0d07c22`; this report's draw tables are configuration `r16_lp3`: LP 5 epochs at 0.0003 on the three output layers, FT 10 epochs at 0.0001 on every head parameter; batch 16 B0268 frames + 16x source replay; geometry dropout 0.3 on the source part; fixed epochs, no model selection on the held-out days. B0268 columns: the model against each rater of the P041 pair, averaged, on the draw's held-out adaptation days (`c2_*_ratio` = model / the raters' pairwise agreement). Source columns: the in-domain validation split; a draw fails when the mean étage or genus F1 drops by more than 0.02.

## Configurations (mean over the k >= 1 draws; held-out days against each rater, averaged)

| Tag | Draws | Étage set after (gain) | Oktas ±1 after (gain) | Genus top-1 after (gain) | Source étage F1 change | Source genus F1 change | Worst source drop | Within the rule |
|---|---|---|---|---|---|---|---|---|
| `base` | 6 | 0.228 (+0.177) | 0.732 (+0.393) | 0.841 (+0.575) | -0.015 | -0.026 | 0.043 | no |
| `lp_only` | 6 | 0.200 (+0.149) | 0.643 (+0.304) | 0.761 (+0.495) | -0.015 | -0.029 | 0.052 | no |
| `r16_ft3` | 6 | 0.211 (+0.160) | 0.574 (+0.235) | 0.545 (+0.278) | -0.010 | -0.022 | 0.038 | no |
| `r16_lp3` | 6 | 0.274 (+0.223) | 0.705 (+0.365) | 0.784 (+0.517) | -0.010 | -0.015 | 0.030 | no |
| `r8` | 6 | 0.236 (+0.185) | 0.723 (+0.383) | 0.830 (+0.564) | -0.016 | -0.025 | 0.044 | no |
| `r8_lp3` | 6 | 0.262 (+0.211) | 0.711 (+0.372) | 0.811 (+0.544) | -0.014 | -0.018 | 0.031 | no |
| `r8_lp3_ft3` | 6 | 0.209 (+0.158) | 0.548 (+0.209) | 0.559 (+0.292) | -0.011 | -0.024 | 0.047 | no |

## Draw `k1_d0` (k = 1: train ['2026-10-18'] / 120 frames; held out ['2026-10-19', '2026-10-21'] / 153 frames; run `20261023T122259Z_adapt_b0268_k1_d0_r16_lp3_667c44e09e`; verdict **source regression**)

| Scope | Metric | Before | After | Change |
|---|---|---|---|---|
| B0268 held-out | n | 153.000 | 153.000 | 0.000 |
| B0268 held-out | etage_set_agreement | 0.026 | 0.157 | 0.131 |
| B0268 held-out | c2_etage_ratio | 0.029 | 0.172 | 0.143 |
| B0268 held-out | oktas_within1 | 0.085 | 0.170 | 0.085 |
| B0268 held-out | c2_oktas_ratio | 0.110 | 0.220 | 0.110 |
| B0268 held-out | oktas_mae | 3.595 | 2.673 | -0.922 |
| B0268 held-out | oktas_bias | 3.595 | 2.660 | -0.935 |
| B0268 held-out | genus_top1_in_set | 0.007 | 0.739 | 0.732 |
| source val | mean_etage_f1 | 0.817 | 0.811 | -0.006 |
| source val | mean_genus_f1 | 0.719 | 0.689 | -0.030 |
| source val | montenegro/c2_etage_ratio | 1.047 | 1.034 | -0.014 |
| source val | montenegro/c2_oktas_ratio | 1.027 | 1.031 | 0.004 |

| Held-out day | étage set before | after | oktas ±1 before | after |
|---|---|---|---|---|
| 2026-10-19 | 0.000 | 0.000 | 0.050 | 0.133 |
| 2026-10-21 | 0.121 | 0.727 | 0.212 | 0.303 |

## Draw `k1_d1` (k = 1: train ['2026-10-19'] / 120 frames; held out ['2026-10-18', '2026-10-21'] / 153 frames; run `20261023T122319Z_adapt_b0268_k1_d1_r16_lp3_f5d431dd31`; verdict **ok**)

| Scope | Metric | Before | After | Change |
|---|---|---|---|---|
| B0268 held-out | n | 153.000 | 153.000 | 0.000 |
| B0268 held-out | etage_set_agreement | 0.072 | 0.307 | 0.235 |
| B0268 held-out | c2_etage_ratio | 0.079 | 0.337 | 0.258 |
| B0268 held-out | oktas_within1 | 0.601 | 0.892 | 0.291 |
| B0268 held-out | c2_oktas_ratio | 0.780 | 1.157 | 0.377 |
| B0268 held-out | oktas_mae | 1.915 | 1.013 | -0.902 |
| B0268 held-out | oktas_bias | 1.379 | -0.229 | -1.608 |
| B0268 held-out | genus_top1_in_set | 0.542 | 0.938 | 0.395 |
| source val | mean_etage_f1 | 0.817 | 0.808 | -0.009 |
| source val | mean_genus_f1 | 0.719 | 0.708 | -0.011 |
| source val | montenegro/c2_etage_ratio | 1.047 | 1.036 | -0.011 |
| source val | montenegro/c2_oktas_ratio | 1.027 | 1.038 | 0.011 |

| Held-out day | étage set before | after | oktas ±1 before | after |
|---|---|---|---|---|
| 2026-10-18 | 0.058 | 0.150 | 0.708 | 0.892 |
| 2026-10-21 | 0.121 | 0.879 | 0.212 | 0.894 |

## Draw `k1_d2` (k = 1: train ['2026-10-21'] / 33 frames; held out ['2026-10-18', '2026-10-19'] / 240 frames; run `20261023T122338Z_adapt_b0268_k1_d2_r16_lp3_de31e40a1c`; verdict **ok**)

| Scope | Metric | Before | After | Change |
|---|---|---|---|---|
| B0268 held-out | n | 240.000 | 240.000 | 0.000 |
| B0268 held-out | etage_set_agreement | 0.029 | 0.029 | 0.000 |
| B0268 held-out | c2_etage_ratio | 0.032 | 0.032 | 0.000 |
| B0268 held-out | oktas_within1 | 0.379 | 0.713 | 0.333 |
| B0268 held-out | c2_oktas_ratio | 0.492 | 0.924 | 0.432 |
| B0268 held-out | oktas_mae | 2.254 | 1.196 | -1.058 |
| B0268 held-out | oktas_bias | 1.913 | 0.417 | -1.496 |
| B0268 held-out | genus_top1_in_set | 0.350 | 0.479 | 0.129 |
| source val | mean_etage_f1 | 0.817 | 0.804 | -0.013 |
| source val | mean_genus_f1 | 0.719 | 0.704 | -0.015 |
| source val | montenegro/c2_etage_ratio | 1.047 | 1.052 | 0.005 |
| source val | montenegro/c2_oktas_ratio | 1.027 | 1.047 | 0.020 |

| Held-out day | étage set before | after | oktas ±1 before | after |
|---|---|---|---|---|
| 2026-10-18 | 0.058 | 0.058 | 0.708 | 0.958 |
| 2026-10-19 | 0.000 | 0.000 | 0.050 | 0.467 |

## Draw `k2_d0` (k = 2: train ['2026-10-18', '2026-10-19'] / 240 frames; held out ['2026-10-21'] / 33 frames; run `20261023T122345Z_adapt_b0268_k2_d0_r16_lp3_babbed0125`; verdict **ok**)

| Scope | Metric | Before | After | Change |
|---|---|---|---|---|
| B0268 held-out | n | 33.000 | 33.000 | 0.000 |
| B0268 held-out | etage_set_agreement | 0.121 | 0.879 | 0.758 |
| B0268 held-out | c2_etage_ratio | 0.133 | 0.965 | 0.832 |
| B0268 held-out | oktas_within1 | 0.212 | 0.924 | 0.712 |
| B0268 held-out | c2_oktas_ratio | 0.275 | 1.199 | 0.924 |
| B0268 held-out | oktas_mae | 4.576 | 1.030 | -3.545 |
| B0268 held-out | oktas_bias | 4.576 | 0.909 | -3.667 |
| B0268 held-out | genus_top1_in_set | 0.000 | 0.788 | 0.788 |
| source val | mean_etage_f1 | 0.817 | 0.799 | -0.019 |
| source val | mean_genus_f1 | 0.719 | 0.704 | -0.014 |
| source val | montenegro/c2_etage_ratio | 1.047 | 1.029 | -0.018 |
| source val | montenegro/c2_oktas_ratio | 1.027 | 1.040 | 0.013 |

| Held-out day | étage set before | after | oktas ±1 before | after |
|---|---|---|---|---|
| 2026-10-21 | 0.121 | 0.879 | 0.212 | 0.924 |

## Draw `k2_d1` (k = 2: train ['2026-10-18', '2026-10-21'] / 153 frames; held out ['2026-10-19'] / 120 frames; run `20261023T122418Z_adapt_b0268_k2_d1_r16_lp3_23c9461740`; verdict **ok**)

| Scope | Metric | Before | After | Change |
|---|---|---|---|---|
| B0268 held-out | n | 120.000 | 120.000 | 0.000 |
| B0268 held-out | etage_set_agreement | 0.000 | 0.033 | 0.033 |
| B0268 held-out | c2_etage_ratio | 0.000 | 0.037 | 0.037 |
| B0268 held-out | oktas_within1 | 0.050 | 0.554 | 0.504 |
| B0268 held-out | c2_oktas_ratio | 0.065 | 0.719 | 0.654 |
| B0268 held-out | oktas_mae | 3.325 | 1.317 | -2.008 |
| B0268 held-out | oktas_bias | 3.325 | 0.642 | -2.683 |
| B0268 held-out | genus_top1_in_set | 0.008 | 0.775 | 0.767 |
| source val | mean_etage_f1 | 0.817 | 0.811 | -0.006 |
| source val | mean_genus_f1 | 0.719 | 0.703 | -0.016 |
| source val | montenegro/c2_etage_ratio | 1.047 | 1.043 | -0.005 |
| source val | montenegro/c2_oktas_ratio | 1.027 | 1.033 | 0.007 |

| Held-out day | étage set before | after | oktas ±1 before | after |
|---|---|---|---|---|
| 2026-10-19 | 0.000 | 0.033 | 0.050 | 0.554 |

## Draw `k2_d2` (k = 2: train ['2026-10-19', '2026-10-21'] / 153 frames; held out ['2026-10-18'] / 120 frames; run `20261023T122440Z_adapt_b0268_k2_d2_r16_lp3_40c5c7dca8`; verdict **ok**)

| Scope | Metric | Before | After | Change |
|---|---|---|---|---|
| B0268 held-out | n | 120.000 | 120.000 | 0.000 |
| B0268 held-out | etage_set_agreement | 0.058 | 0.237 | 0.179 |
| B0268 held-out | c2_etage_ratio | 0.064 | 0.261 | 0.197 |
| B0268 held-out | oktas_within1 | 0.708 | 0.975 | 0.267 |
| B0268 held-out | c2_oktas_ratio | 0.919 | 1.265 | 0.346 |
| B0268 held-out | oktas_mae | 1.183 | 0.658 | -0.525 |
| B0268 held-out | oktas_bias | 0.500 | -0.242 | -0.742 |
| B0268 held-out | genus_top1_in_set | 0.692 | 0.983 | 0.292 |
| source val | mean_etage_f1 | 0.817 | 0.813 | -0.005 |
| source val | mean_genus_f1 | 0.719 | 0.714 | -0.004 |
| source val | montenegro/c2_etage_ratio | 1.047 | 1.040 | -0.007 |
| source val | montenegro/c2_oktas_ratio | 1.027 | 1.036 | 0.009 |

| Held-out day | étage set before | after | oktas ±1 before | after |
|---|---|---|---|---|
| 2026-10-18 | 0.058 | 0.237 | 0.708 | 0.975 |

## Draw `k0_d0` (k = 3: train ['2026-10-18', '2026-10-19', '2026-10-21'] / 273 frames; held out none / 0 frames; run `20261023T123549Z_adapt_b0268_final_r16_lp3_76502c7936`; verdict **ok**)

| Scope | Metric | Before | After | Change |
|---|---|---|---|---|
| source val | mean_etage_f1 | 0.817 | 0.805 | -0.012 |
| source val | mean_genus_f1 | 0.719 | 0.701 | -0.018 |
| source val | montenegro/c2_etage_ratio | 1.047 | 1.029 | -0.018 |
| source val | montenegro/c2_oktas_ratio | 1.027 | 1.029 | 0.002 |

