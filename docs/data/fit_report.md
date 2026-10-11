# Fit sign-off (P081)

Train / val at the best epoch per head and dataset (global heads: étage and genus macro-F1 on the fixed train subset vs val), the PLAN §4 verdict, and calibration on validation rows. Validation only; the locked test sets stay untouched.

## Global heads (P073 candidate)

| Head | Dataset | Train | Val | Gap | Verdict |
|---|---|---|---|---|---|
| etage | almeria | 0.924 | 0.775 | +0.149 | ok (gap 0.15) |
| etage | ccsn | 0.961 | 0.802 | +0.159 | overfit-leaning (gap 0.16) |
| etage | mgcd | 1.000 | 1.000 | +0.000 | ok (gap 0.00) |
| etage | montenegro | 0.776 | 0.692 | +0.083 | ok (gap 0.08) |
| genus | ccsn | 0.876 | 0.626 | +0.250 | overfit-leaning (gap 0.25) |
| genus | mgcd | 0.998 | 0.812 | +0.186 | overfit (gap 0.19, val fell after epoch 11) |

Calibration (val): étage ECE 0.013 on 1352 rows, genus ECE 0.059 on 895 fully-labelled rows, oktas top-class ECE 0.061.

## Dense heads (20261011T095907Z_dense_sky_layer_s0_dcd14f4833)

Val mIoU curve: 0.839 0.863 0.881 0.882 0.880 0.886 0.892 0.892 0.896 0.899 0.902 0.903 0.904 0.905 0.904; best epoch 14 (0.905); train loss 1.148 → -0.887; val fell after epoch 14.

## joint (20261011T104158Z_joint_joint_long_s0_865b5aa8f4)

Headline curve: 0.688 0.719 0.719 0.732 0.726 0.735 0.735 0.729 0.731 0.730 0.727 0.729; best epoch 7.

## lora (20261011T113235Z_lora_lora_lowhead_s0_e013bf743c)

Headline curve: 0.843 0.838 0.837; best epoch 1.

## CBH head (P075 candidate)

80 % interval coverage on val: 0.917 raw; with a temperature cross-fitted over the validation days (0.49): 0.802, MAE of the median after scaling 580 m. The proper calibration split is P089's.
