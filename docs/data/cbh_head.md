# CBH head on cached DINOv3 tokens (P067)

P047 pairs, 10-minute cadence, train -> val days (test locked); GeometryEncoder + CbhHead on the 8 x 8 pooled token grid with the contract's ray map and Sun meta (geometry dropout 0.3 in training). Skill = 1 - MAE / 1168 m (train-median climatology). 'geometry zeroed' scores the same weights with the ray map and meta set to the contract's unknown state. The P058 baselines are listed for reference.

| Phase | Model | Variant | MAE (m) | Bias (m) | Skill | ≤500 m | CRPS | 80 % coverage | Cloud bal. acc | Top-bin share pred / true | Entropy ratio | Best epoch | OLDLR MAE / skill | WESTE MAE / skill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P067 | cbh_head | geometry zeroed | 566 | +125 | +0.507 | 0.70 | 0.0273 | 0.94 | 0.921 | 0.26 / 0.19 | 0.91 | 8 | - | - |
| P067 | cbh_head | geometry zeroed (no geometry dropout) | 765 | -528 | +0.333 | 0.62 | 0.0386 | 0.91 | 0.878 | 0.39 / 0.19 | 0.57 | 4 | - | - |
| P067 | cbh_head | with geometry | 566 | +125 | +0.506 | 0.69 | 0.0273 | 0.92 | 0.926 | 0.25 / 0.19 | 0.95 | 8 | - | - |
| P067 | cbh_head | with geometry (no geometry dropout) | 551 | +80 | +0.520 | 0.70 | 0.0269 | 0.94 | 0.924 | 0.26 / 0.19 | 0.95 | 4 | - | - |
| P075 | cbh_head | geometry zeroed (+synthetic views) | 541 | +120 | +0.528 | 0.70 | 0.0269 | 0.94 | 0.919 | 0.30 / 0.19 | 0.91 | 9 | - | - |
| P075 | cbh_head | geometry zeroed (OLDLR only; WESTE held out) | 615 | +93 | +0.466 | 0.66 | 0.0310 | 0.92 | 0.876 | 0.24 / 0.19 | 0.91 | 9 | - | - |
| P075 | cbh_head | geometry zeroed (day-weighted) | 582 | +225 | +0.493 | 0.69 | 0.0285 | 0.94 | 0.919 | 0.26 / 0.19 | 0.94 | 8 | - | - |
| P075 | cbh_head | geometry zeroed (p-drop 0.15) | 557 | -150 | +0.514 | 0.71 | 0.0279 | 0.95 | 0.912 | 0.34 / 0.19 | 0.80 | 4 | - | - |
| P075 | cbh_head | geometry zeroed [P075 base] | 572 | +193 | +0.501 | 0.70 | 0.0277 | 0.94 | 0.928 | 0.26 / 0.19 | 0.90 | 7 | - | - |
| P075 | cbh_head | geometry zeroed [P075 base] | 559 | +140 | +0.512 | 0.70 | 0.0283 | 0.95 | 0.922 | 0.24 / 0.19 | 0.98 | 5 | - | - |
| P075 | cbh_head | geometry zeroed [P075 base] | 566 | +125 | +0.507 | 0.70 | 0.0273 | 0.94 | 0.921 | 0.26 / 0.19 | 0.91 | 8 | - | - |
| P075 | cbh_head | with geometry (+synthetic views) | 538 | +133 | +0.531 | 0.70 | 0.0265 | 0.92 | 0.924 | 0.28 / 0.19 | 0.93 | 9 | 554 / +0.517 | 523 / +0.544 |
| P075 | cbh_head | with geometry (OLDLR only; WESTE held out) | 610 | +100 | +0.470 | 0.66 | 0.0307 | 0.89 | 0.858 | 0.22 / 0.19 | 0.95 | 9 | 582 / +0.494 | 636 / +0.448 |
| P075 | cbh_head | with geometry (day-weighted) | 574 | +197 | +0.499 | 0.69 | 0.0276 | 0.93 | 0.919 | 0.26 / 0.19 | 0.97 | 8 | 621 / +0.459 | 530 / +0.538 |
| P075 | cbh_head | with geometry (p-drop 0.15) | 560 | +101 | +0.511 | 0.71 | 0.0272 | 0.94 | 0.922 | 0.22 / 0.19 | 0.95 | 4 | 600 / +0.477 | 523 / +0.544 |
| P075 | cbh_head | with geometry [P075 base] | 546 | +114 | +0.524 | 0.70 | 0.0262 | 0.93 | 0.930 | 0.27 / 0.19 | 0.94 | 7 | 566 / +0.507 | 528 / +0.540 |
| P075 | cbh_head | with geometry [P075 base] | 558 | +127 | +0.513 | 0.71 | 0.0282 | 0.93 | 0.926 | 0.19 / 0.19 | 1.01 | 5 | 577 / +0.497 | 541 / +0.528 |
| P075 | cbh_head | with geometry [P075 base] | 566 | +125 | +0.506 | 0.69 | 0.0273 | 0.92 | 0.926 | 0.25 / 0.19 | 0.95 | 8 | 607 / +0.471 | 528 / +0.540 |
| P079 | cbh_head | geometry zeroed (train days 0.1) | 766 | -219 | +0.321 | 0.60 | 0.0429 | 0.85 | 0.903 | 0.42 / 0.19 | 0.65 | 9 | - | - |
| P079 | cbh_head | geometry zeroed (train days 0.25) | 672 | -226 | +0.411 | 0.64 | 0.0349 | 0.93 | 0.916 | 0.38 / 0.19 | 0.78 | 4 | - | - |
| P079 | cbh_head | geometry zeroed (train days 0.5) | 614 | +132 | +0.474 | 0.64 | 0.0313 | 0.93 | 0.925 | 0.48 / 0.19 | 0.67 | 5 | - | - |
| P079 | cbh_head | with geometry (train days 0.1) | 756 | -184 | +0.330 | 0.60 | 0.0420 | 0.82 | 0.892 | 0.38 / 0.19 | 0.68 | 9 | 772 / +0.315 | 740 / +0.343 |
| P079 | cbh_head | with geometry (train days 0.25) | 661 | -90 | +0.420 | 0.63 | 0.0341 | 0.92 | 0.919 | 0.20 / 0.19 | 0.89 | 4 | 698 / +0.388 | 626 / +0.451 |
| P079 | cbh_head | with geometry (train days 0.5) | 580 | +24 | +0.503 | 0.64 | 0.0294 | 0.91 | 0.928 | 0.42 / 0.19 | 0.81 | 5 | 615 / +0.473 | 548 / +0.531 |
| P058 | climatology_median | - | 1147 | -312 | +0.000 | 0.27 | - | - | - | - | - | - | - | - |
| P058 | climatology_month_hour | - | 1131 | -330 | +0.014 | 0.32 | - | - | - | - | - | - | - | - |
| P058 | etage_lookup | - | 1124 | -767 | +0.020 | 0.44 | - | - | 0.656 | - | - | - | - | - |
| P058 | ridge_dinov3_vits16_512 | - | 699 | +34 | +0.390 | 0.61 | - | - | 0.909 | - | - | - | - | - |
| P058 | cnn_resnet18_logheight | - | 724 | -42 | +0.368 | 0.63 | - | - | - | - | - | - | - | - |

Rows in `data/cbh_head.parquet`; curves in the run's `metrics.jsonl`.
