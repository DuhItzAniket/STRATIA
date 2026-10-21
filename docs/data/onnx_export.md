# ONNX export (P096)

`models/stratia-1.0.0-demo/stratia.onnx`: 94.9 MB, SHA-256 `d3c918ecb105b807651f0f31d5e11bbff5f5e97715d799ab531097694c84a547`, opset 17, dynamic batch, FP32. Model card `models/stratia-1.0.0-demo/model_card.json` (stratia-contract v1.0, model `stratia` 1.0.0-demo, git `98c756df80`), also embedded in the ONNX metadata under `stratia_model_card`.

Graph: frozen DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`) at 512 px; dense heads on the 32 x 32 token grid; global and cloud-base heads with their own geometry encoders on the grid average-pooled to 8 x 8 (as trained); the contract's 32 x 32 ray map is pooled to 8 x 8 in the graph; P089 temperatures and OOD statistics are folded in (`calibration.folded_into_graph = true`), so hosts apply nothing but the threshold.

## Provenance

| Part | Run | Validation summary |
|---|---|---|
| global | `20261011T125754Z_global_indomain_e2s_s0_27d0d07c22` | best 0.817, best_epoch 11, epochs 17, val/montenegro/c2_etage_ratio 1.047, val/montenegro/c2_oktas_ratio 1.027, val/mean_etage_f1 0.817, val/mean_genus_f1 0.719 |
| cbh | `20261011T101553Z_cbh_head_s0_[P075_base]_e286609cd2` | val_mae_m 566.100, val_skill 0.506, val_crps 0.027, val_coverage_80 0.917, null_geometry_mae_m 565.821, best_epoch 8 |
| dense | `20261011T150404Z_dense_sky_layer_s0_dcd14f4833` | val_miou 0.904, best_epoch 14, clear_sky_cloud_fraction 0.290, b0268_cloud_fraction 0.448, val_layer_miou 0.558 |

Calibration (P089): temperatures genus 4.4913, étage 1.0, oktas 1.0, CBH 1.2394; OOD threshold 1.4569; CBH conformal radii 80 % 0.327, 95 % 0.438 (CDF units).

## Parity (PyTorch FP32 vs ONNX Runtime CPU)

| Output | Shape (batch of 4) | Random inputs: max abs | max rel | Real frames: max abs | max rel |
|---|---|---|---|---|---|
| `genus_logits` | (4, 12) | 9.30e-06 | 2.25e-06 | 5.01e-06 | 1.53e-06 |
| `etage_logits` | (4, 3) | 3.31e-05 | 3.63e-06 | 1.31e-05 | 1.98e-06 |
| `oktas_probs` | (4, 9) | 2.62e-06 | 3.15e-06 | 7.90e-07 | 2.70e-06 |
| `sky_parse_logits` | (4, 4, 128, 128) | 1.31e-04 | 1.01e-05 | 1.81e-04 | 1.01e-05 |
| `layer_logits` | (4, 3, 128, 128) | 1.52e-04 | 5.47e-05 | 1.46e-04 | 1.95e-05 |
| `cbh_probs` | (4, 25) | 8.05e-07 | 5.33e-06 | 2.98e-07 | 2.03e-06 |
| `ood_score` | (4, 1) | 3.10e-06 | 9.47e-07 | 1.79e-06 | 1.20e-06 |

Contract check (names, shapes, finite values, probability sums): passed. Half of every parity batch has an all-zero ray map and meta (unknown geometry).

ONNX Runtime CPU latency, batch 1, median of 5: 172 ms (Intel64 Family 6 Model 183 Stepping 1, GenuineIntel).

## Ray-map pooling

The global and CBH heads were trained on ray maps computed directly at the 8 x 8 patch; the graph pools the contract's 32 x 32 map instead (mean of the valid unit directions, re-normalised; valid when at least half of the sub-patches are). On the nominal B0268 camera (zenith pose, Sun at 45° zenith): 64 / 64 valid patches (direct / pooled), 0 flag disagreements, direction error mean 0.031°, max 0.048° (the CBH prior's sigma is 20°).

## Preview on B0268 frames (not an evaluation)

Frames from `C:/Users/Luikz/Downloads/STRATIA/DATA/B0268/BLR01/2026-10-19` through the exported graph with the nominal B0268 camera, a zenith pose and the sidecar's Sun. The model has never seen this camera; P083 (zero-shot, with labels) is the evaluation. Shown only to confirm the graph produces sensible, finite outputs on real frames.

| Frame | OOD score | Uncertain | Oktas | Étage p(low, mid, high) | Genus top-3 | p(cloud overhead) | CBH median | Sky-parse fractions |
|---|---|---|---|---|---|---|---|---|
| BLR01_20261019T093909_479Z.jpg | 1.485 | yes | 8 | [0.0, 0.53, 0.9] | [('Ns', 0.56), ('Cs', 0.52), ('As', 0.4)] | 0.94 | 6356 m | {'invalid': 0.0, 'sky': 0.692, 'cloud': 0.308, 'sun_glare': 0.0} |
| BLR01_20261019T095839_791Z.jpg | 1.073 | no | 8 | [0.0, 0.9, 0.79] | [('Ns', 0.66), ('Cs', 0.53), ('Cc', 0.48)] | 0.99 | 2141 m | {'invalid': 0.0, 'sky': 0.655, 'cloud': 0.345, 'sun_glare': 0.0} |
| BLR01_20261019T101840_031Z.jpg | 1.216 | no | 8 | [0.01, 0.81, 0.88] | [('Ns', 0.63), ('Cs', 0.51), ('Ac', 0.4)] | 0.99 | 2141 m | {'invalid': 0.0, 'sky': 0.41, 'cloud': 0.59, 'sun_glare': 0.0} |
| BLR01_20261019T103840_008Z.jpg | 1.154 | no | 6 | [0.01, 0.94, 0.57] | [('Ns', 0.64), ('Ac', 0.48), ('Cs', 0.47)] | 0.99 | 4256 m | {'invalid': 0.0, 'sky': 0.391, 'cloud': 0.609, 'sun_glare': 0.0} |

## Licences

- Apache-2.0 (STRATIA code and head weights)
- DINOv3 License (backbone weights facebook/dinov3-vits16-pretrain-lvd1689m): Built with DINOv3
- training datasets keep their own licences (docs/data/data_card.md)
