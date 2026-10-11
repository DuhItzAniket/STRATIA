# CNN baselines (P052 / P053)

ImageNet-pretrained timm models at 224 px on the P022 image cache; `grouped` = P038 in-domain split (train -> val, test locked); `legacy_random` = a per-image random split of the same pool with the same per-class val counts (one draw per seed), its val scored whole and split into *leaked* images (a near-duplicate unit member is in train) and *clean* ones. Mean ± std over seeds where more than one seed ran.

| Phase | Arch | Source | Protocol | Recipe | Seeds | Val macro-F1 | Val acc | Leaked val: n, F1 | Clean val F1 | Train-val gap | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P052 | mobilenetv3_large | ccsn | grouped | legacy | 3 | 0.588 ± 0.018 | 0.595 | 0 | 0.588 | +0.261 | 10.0 |
| P052 | mobilenetv3_large | ccsn | legacy_random | legacy | 3 | 0.511 ± 0.029 | 0.525 | 24, 0.146 | 0.543 | +0.280 | 8.7 |
| P053 | convnext_tiny | ccsn | grouped | modern | 3 | 0.595 ± 0.010 | 0.616 | 0 | - | +0.270 | 8.3 |
| P053 | convnext_tiny | mgcd | grouped | modern | 3 | 0.793 ± 0.016 | 0.824 | 0 | - | +0.174 | 7.0 |
| P053 | convnext_tiny | montenegro | grouped | modern | 3 | 0.695 ± 0.024 | 0.824 | 0 | - | +0.138 | 9.7 |
| P053 | convnext_tiny | swimcat | grouped | modern | 3 | 1.000 ± 0.000 | 1.000 | 0 | - | -0.031 | 4.0 |
| P053 | resnet50 | ccsn | grouped | modern | 3 | 0.527 ± 0.009 | 0.556 | 0 | - | +0.088 | 8.7 |
| P053 | resnet50 | mgcd | grouped | modern | 3 | 0.768 ± 0.026 | 0.782 | 0 | - | +0.210 | 6.3 |
| P053 | resnet50 | montenegro | grouped | modern | 3 | 0.599 ± 0.011 | 0.821 | 0 | - | +0.073 | 8.3 |
| P053 | resnet50 | swimcat | grouped | modern | 3 | 1.000 ± 0.000 | 1.000 | 0 | - | -0.025 | 6.7 |

Verdicts per run (PLAN §4 fit reading) are in `data/baseline_cnn.parquet`; curves in each run's `metrics.jsonl`.
