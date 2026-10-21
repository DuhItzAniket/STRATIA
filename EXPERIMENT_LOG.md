# Experiment Log

One entry per experiment run that informs a decision. The machine-readable record is `runs/registry.csv` (git SHA, config hash, data hash, metrics); this log adds the human reasoning.

| Date | Run ID | Phase | Question | Result (val unless stated) | Decision |
|---|---|---|---|---|---|
| 2026-10-04 | 20261004T112119Z_repro_check_a/b_b011a8395a | P004 | Are two runs with one config bit-identical on this GPU? | Yes: identical loss/accuracy for all 3 epochs (acc 0.5688 / 0.86945 / 0.91775) | Deterministic mode is adequate; still report mean ± std over seeds for real models |
| 2026-10-21 | 20261021T144545Z_onnx_export_8162d41fc1 | P096 | Does the Gate G4 model survive the ONNX export unchanged? | Parity vs ONNX Runtime ≤ 1.5e-4 absolute on every output (random inputs and four B0268 frames), contract check passed, 172 ms per frame on the laptop CPU, ray-map pooling error ≤ 0.05° | Ship `models/stratia-1.0.0-demo/` to CloudScope Stage F; INT8 and the locked-test re-export wait for P095/P097 |
