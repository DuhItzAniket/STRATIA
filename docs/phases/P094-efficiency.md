# P094 — Efficiency

Status: DONE     Date: 2026-11-05     Commit: (this commit)

## Objective
Plan (Stage J, P094): parameters, FLOPs and latency of the deployed graph on the laptop GPU, the CPU and inside
CloudScope; the Raspberry Pi 5 row of the paper's efficiency table.

## Inputs / dependencies
- `models/stratia-1.0.1-demo/stratia.onnx` and its card (the graph CloudScope runs); `stratia.export.onnx_model`
  (`load_trained`, `example_inputs`) for the PyTorch side; `torch.utils.flop_counter.FlopCounterMode`;
  `cloudscope-aibench` from the CloudScope Debug build (`--provider all --json`, the rehearsal frame
  `BLR01_20261025T104657_734Z.jpg` as the 16 MP input).
- `scripts/efficiency.py` -> `data/efficiency.json`, `docs/data/efficiency.md`; `scripts/reproduce_tables.py` ->
  `paper/tables/tab7_efficiency.{tex,md}` and `\nParamsM`, `\nHeadsParamsM`, `\nGFlops`, `\nLatencyGpu`,
  `\nLatencyCpu`, `\nLatencyDml` in `paper/numbers.tex`.

## Work log
1. Parameters: 23.36 M in total, of which the frozen DINOv3 ViT-S/16 backbone 21.60 M and 1.76 M trainable above
   it (global heads 0.996 M, dense heads 0.355 M, CBH head 0.206 M, the two SkyRay encoders 0.101 M each; the OOD
   score keeps 0.263 M of buffers).
2. FLOPs for one 512 x 512 frame: 45.7 G, of which the backbone 44.3 G; everything above it costs 1.40 G.
3. Latency (median / p95 over repeated runs after warm-up, run `20261105T141627Z_efficiency_578b90289f`):
   PyTorch fp32 on the RTX 4050 Laptop GPU 19.5 / 23.7 ms at batch 1, 15.0 / 16.2 ms with fp16 autocast, 16.5 ms
   per frame at batch 8; ONNX Runtime CPU 144.1 / 149.4 ms; inside CloudScope (`cloudscope-aibench`, ONNX Runtime
   1.24) 29.2 / 33.0 ms through DirectML and 148.7 / 153.7 ms on the CPU provider, plus 29 ms of host preprocessing
   (the full-frame INTER_AREA resize of the 16 MP B0268 frame) in both cases. The desktop application therefore
   runs the model at about 17 frames per second end to end on the laptop GPU, which is far above the camera's
   capture cadence.
4. The FLOP counter's module tracker asserts on grad-requiring leaf inputs under `no_grad` (the heads' learned
   queries); the script sets `requires_grad_(False)` on every parameter before counting.

## Deviations from the plan
- The Raspberry Pi 5 row is left blank: no Pi 5 on hand, so the edge student (P095) and INT8 quantisation
  (P096's second half) remain deferred and the paper says "not measured" rather than estimating.

## Outputs
- `data/efficiency.json`, `docs/data/efficiency.md`, `paper/tables/tab7_efficiency.{tex,md}`, `paper/numbers.tex`.
