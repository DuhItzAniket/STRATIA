# Efficiency (P094)

The deployed graph (`models\stratia-1.0.1-demo\stratia.onnx`, 512 x 512 input), run `20261105T141627Z_efficiency_578b90289f` on LEGION.

## Parameters

| Component | Parameters |
|---|---|
| backbone (DINOv3 ViT-S/16, frozen) | 21.597 M |
| SkyRay encoder, global | 0.101 M |
| global heads | 0.996 M |
| SkyRay encoder, CBH | 0.101 M |
| CBH head | 0.206 M |
| dense heads | 0.355 M |
| OOD score (buffers) | 0.263 M |
| total trainable above the backbone | 1.760 M |
| total | 23.356 M |

FLOPs per frame (one forward pass at 512 x 512): **45.7 GFLOPs**, of which the backbone 44.3 GFLOPs; everything above the backbone costs 1.40 GFLOPs.

## Latency

| Setting | Median ms | p95 ms | Runs | Note |
|---|---|---|---|---|
| pytorch fp32, RTX 4050, batch 1 | 19.5 | 23.7 | 20 |  |
| pytorch autocast fp16, RTX 4050, batch 1 | 15.0 | 16.2 | 20 |  |
| pytorch fp32, RTX 4050, batch 8 (per frame) | 16.5 | 16.5 | 10 |  |
| onnx runtime CPU, batch 1 | 144.1 | 149.4 | 10 |  |
| CloudScope ONNX Runtime DirectML (NVIDIA GeForce RTX 4050 Laptop GPU), batch 1 | 29.2 | 33.0 | 20 | preprocess 29 ms (host resize of a 16 MP frame) |
| CloudScope ONNX Runtime CPU, batch 1 | 148.7 | 153.7 | 20 | preprocess 29 ms (host resize of a 16 MP frame) |
| Raspberry Pi 5 | — | — | — | not measured: no Raspberry Pi 5 on hand |

GPU: NVIDIA GeForce RTX 4050 Laptop GPU; CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel. The CloudScope rows come from `cloudscope-aibench` (ONNX Runtime 1.24 inside the desktop application) and are the numbers the demo runs at; the PyTorch rows are the research model without export. NFR-PERF-05 (CloudScope) asks for the latency targets here.
