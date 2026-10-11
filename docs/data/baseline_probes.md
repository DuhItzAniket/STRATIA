# Frozen-feature probe baselines (P054 / P055)

Linear probe (standardised features, multinomial logistic regression, C chosen on val) and cosine k-NN (k = 20) on the pooled embedding of each frozen backbone. In-domain: grouped train -> grouped val, macro-F1 over native classes. LODO étage: train on the other sources' train + val rows, score on the held-out source's LODO val units, macro-F1 over the low / mid / high binary labels restricted to the columns with val support (MGCD's val has no mid-étage rows: scored on low and high; false positives in unsupported columns are in the parquet as `*_unsupported_fp`). Test sets untouched.

## In-domain (val)

| Backbone | Size | ccsn lin / kNN | mgcd lin / kNN | swimcat lin / kNN | montenegro lin / kNN | Mean lin |
|---|---|---|---|---|---|---|
| clip_vitb16 | 224 | 0.597 / 0.540 | 0.758 / 0.486 | 1.000 / 0.957 | 0.627 / 0.556 | 0.746 |
| dinov2_base | 224 | 0.580 / 0.453 | 0.765 / 0.569 | 1.000 / 0.964 | 0.607 / 0.484 | 0.738 |
| dinov2_small | 224 | 0.570 / 0.411 | 0.810 / 0.546 | 1.000 / 0.941 | 0.573 / 0.495 | 0.738 |
| dinov3_vitb16 | 224 | 0.584 / 0.523 | 0.825 / 0.619 | 1.000 / 0.977 | 0.651 / 0.517 | 0.765 |
| dinov3_vitl16 | 224 | 0.646 / 0.558 | 0.786 / 0.669 | 1.000 / 0.956 | 0.621 / 0.542 | 0.763 |
| dinov3_vits16 | 224 | 0.543 / 0.506 | 0.778 / 0.596 | 1.000 / 0.956 | 0.581 / 0.511 | 0.726 |
| dinov3_vits16 | 512 | 0.586 / 0.561 | 0.831 / 0.609 | 1.000 / 1.000 | 0.600 / 0.538 | 0.754 |
| siglip2_b16 | 224 | 0.594 / 0.560 | 0.762 / 0.525 | 1.000 / 0.966 | 0.614 / 0.509 | 0.743 |

## Leave-one-dataset-out, étage (LODO val)

| Backbone | Size | ccsn lin / kNN | mgcd lin / kNN | montenegro lin / kNN | almeria lin / kNN | Mean lin |
|---|---|---|---|---|---|---|
| clip_vitb16 | 224 | 0.282 / 0.194 | 0.478 / 0.244 | 0.448 / 0.192 | 0.278 / 0.285 | 0.371 |
| dinov2_base | 224 | 0.468 / 0.290 | 0.908 / 0.273 | 0.441 / 0.239 | 0.435 / 0.302 | 0.563 |
| dinov2_small | 224 | 0.327 / 0.354 | 0.739 / 0.248 | 0.295 / 0.297 | 0.239 / 0.278 | 0.400 |
| dinov3_vitb16 | 224 | 0.391 / 0.538 | 0.995 / 0.505 | 0.516 / 0.491 | 0.429 / 0.324 | 0.583 |
| dinov3_vitl16 | 224 | 0.501 / 0.478 | 0.924 / 0.664 | 0.270 / 0.301 | 0.381 / 0.397 | 0.519 |
| dinov3_vits16 | 224 | 0.401 / 0.397 | 0.974 / 0.567 | 0.403 / 0.289 | 0.312 / 0.306 | 0.522 |
| dinov3_vits16 | 512 | 0.441 / 0.421 | 0.731 / 0.769 | 0.341 / 0.409 | 0.227 / 0.305 | 0.435 |
| siglip2_b16 | 224 | 0.379 / 0.280 | 0.953 / 0.162 | 0.527 / 0.400 | 0.256 / 0.343 | 0.529 |

Run ids are in `data/baseline_probes.parquet` and `runs/registry.csv` (phase P054/P055).
