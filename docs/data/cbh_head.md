# CBH head on cached DINOv3 tokens (P067)

P047 pairs, 10-minute cadence, train -> val days (test locked); GeometryEncoder + CbhHead on the 8 x 8 pooled token grid with the contract's ray map and Sun meta (geometry dropout 0.3 in training). Skill = 1 - MAE / 1147 m (train-median climatology). 'geometry zeroed' scores the same weights with the ray map and meta set to the contract's unknown state. The P058 baselines are listed for reference.

| Phase | Model | Variant | MAE (m) | Bias (m) | Skill | ≤500 m | CRPS | 80 % coverage | Cloud bal. acc | Top-bin share pred / true | Entropy ratio | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P067 | cbh_head | geometry zeroed | 566 | +125 | +0.507 | 0.70 | 0.0273 | 0.94 | 0.921 | 0.26 / 0.19 | 0.91 | 8 |
| P067 | cbh_head | geometry zeroed (no geometry dropout) | 765 | -528 | +0.333 | 0.62 | 0.0386 | 0.91 | 0.878 | 0.39 / 0.19 | 0.57 | 4 |
| P067 | cbh_head | with geometry | 566 | +125 | +0.506 | 0.69 | 0.0273 | 0.92 | 0.926 | 0.25 / 0.19 | 0.95 | 8 |
| P067 | cbh_head | with geometry (no geometry dropout) | 551 | +80 | +0.520 | 0.70 | 0.0269 | 0.94 | 0.924 | 0.26 / 0.19 | 0.95 | 4 |
| P058 | climatology_median | - | 1147 | -312 | +0.000 | 0.27 | - | - | - | - | - | - |
| P058 | climatology_month_hour | - | 1131 | -330 | +0.014 | 0.32 | - | - | - | - | - | - |
| P058 | etage_lookup | - | 1124 | -767 | +0.020 | 0.44 | - | - | 0.656 | - | - | - |
| P058 | ridge_dinov3_vits16_512 | - | 699 | +34 | +0.390 | 0.61 | - | - | 0.909 | - | - | - |
| P058 | cnn_resnet18_logheight | - | 724 | -42 | +0.368 | 0.63 | - | - | - | - | - | - |

Rows in `data/cbh_head.parquet`; curves in the run's `metrics.jsonl`.
