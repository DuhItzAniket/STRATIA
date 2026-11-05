**Latency per frame (median / p95 of repeated runs) of the deployed graph; 23.4 M parameters, 46 GFLOPs at 512 x 512.**

| Setting | Median ms | p95 ms |
|---|---|---|
| pytorch fp32, RTX 4050, batch 1 | 19.5 | 23.7 |
| pytorch autocast fp16, RTX 4050, batch 1 | 15.0 | 16.2 |
| pytorch fp32, RTX 4050, batch 8 (per frame) | 16.5 | 16.5 |
| onnx runtime CPU, batch 1 | 144.1 | 149.4 |
| CloudScope ONNX Runtime DirectML (NVIDIA GeForce RTX 4050 Laptop GPU), batch 1 | 29.2 | 33.0 |
| CloudScope ONNX Runtime CPU, batch 1 | 148.7 | 153.7 |
| Raspberry Pi 5 | – | – |
