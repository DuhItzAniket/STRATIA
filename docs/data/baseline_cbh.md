# Cloud-base-height baselines (P058)

P047 pairs at a 10-minute cadence: 14,366 train / 2,193 val pairs, 1,368 cloudy val pairs scored on the lowest base in metres. Skill = 1 - MAE / MAE of the train-median climatology (1147 m). Test days locked.

| Baseline | n | MAE (m) | Bias (m) | RMSD (m) | Skill | ≤250 m | ≤500 m | ≤1000 m | MAE 0-1000 | MAE 1000-2000 | MAE 2000-4000 | MAE 4000-8000 | MAE 8000-20000 | Cloud presence bal. acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| climatology_median | 1,368 | 1147 | -312 | 1898 | +0.000 | 0.15 | 0.27 | 0.67 | 1052 | 360 | 916 | 3515 | 7883 | - |
| climatology_month_hour | 1,368 | 1131 | -330 | 1931 | +0.014 | 0.14 | 0.32 | 0.70 | 818 | 475 | 911 | 3506 | 8104 | - |
| etage_lookup | 1,368 | 1124 | -767 | 1975 | +0.020 | 0.26 | 0.44 | 0.68 | 451 | 359 | 1495 | 3976 | 6852 | 0.656 |
| ridge_dinov3_vits16_512 | 1,368 | 699 | +34 | 1141 | +0.390 | 0.37 | 0.61 | 0.78 | 303 | 486 | 853 | 1859 | 2903 | 0.909 |
| cnn_resnet18_logheight | 1,368 | 724 | -42 | 1274 | +0.368 | 0.39 | 0.63 | 0.80 | 322 | 468 | 816 | 2247 | 3171 | - |

Zero-shot geometry foundation models (DA3, MoGe-2) are not run: they mask the sky by design and return no cloud geometry (research design §1). The ceilometer-supervised CNN is the reference of criterion C3.
