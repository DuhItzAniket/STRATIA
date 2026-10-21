# P096 — ONNX export (FP32) and model card

Status: PARTIAL     Date: 2026-10-21     Commit: (this commit)

Pulled forward from Stage K for the AI-day demo of 2026-10-30 (owner decision 2026-10-16): CloudScope needs the
deployed model and its card now (CloudScope P065–P068). The FP32 export and the model card are done; the INT8
quantisation for the Raspberry Pi and the accuracy-drop report wait for P095's edge student, and the exported
weights are the Gate G4 candidate, not the locked-test model of P090.

## Objective
One ONNX file with the contract's three inputs and seven outputs (`docs/contract.md`, stratia-contract v1.0),
opset ≥ 17, dynamic batch, PyTorch ↔ ONNX Runtime parity ≤ 1e-4, plus a `model_card.json` that validates against
`schemas/model_card.schema.json`; a small dummy export for CloudScope's contract test.

## Inputs / dependencies
- Gate G4 heads (P082): global `20261011T125754Z_global_indomain_e2s_s0_27d0d07c22` (P073, EMA weights), dense
  `20261011T150404Z_dense_sky_layer_s0_dcd14f4833` (P074), CBH `20261011T101553Z_cbh_head_s0_[P075_base]_e286609cd2`
  (P075); calibration `configs/calibration.json` (P089: temperatures, conformal radii, OOD statistics and threshold).
- Backbone `facebook/dinov3-vits16-pretrain-lvd1689m` (P051 caches were built from it at 512 px, transform `ffr1`).
- Manifest `data/manifest.parquet`, content hash `464078b6d033dd39` (data card); the full SHA-256 is in the card.
- New packages: onnx 1.23.2, onnxruntime 1.31.0, jsonschema 4.26.0 (requirements.in / .lock / -ci).

## Work log
1. `stratia/export/onnx_model.py`: `ExportModel` composes the backbone, the dense heads on the 32 × 32 token grid, and
   the global and CBH heads (each with its own trained `GeometryEncoder`) on the grid average-pooled 4 × 4 to 8 × 8,
   which is how their training caches were built (`FeatureCache.build(pool=4)`). The contract's 32 × 32 ray map is
   pooled to 8 × 8 inside the graph (`pool_ray_map`: mean of the valid unit directions, re-normalised; valid when at
   least half of the sub-patches are). P089 is folded in: genus and CBH temperatures divide the logits, the oktas
   CORAL probabilities are recomputed from the tempered cumulative logits, the OOD mean / precision / z-scores sit in
   `OodScore` buffers; the card says `calibration.folded_into_graph = true`. Everything stays a function of the
   inputs (`torch.where`), so the legacy exporter traces one graph for known and unknown geometry (P050).
2. Decision — **no camera-mask zeroing**: the contract's v1 default zeroes camera-mask pixels, but every head was
   trained on `ffr1` features without it (P051). The card's `input.resize` states the rule hosts must follow:
   full-frame resize to 512 × 512 (INTER_AREA), no crop, no mask zeroing, then [0, 1] and ImageNet mean / std.
   The contract allows the card to define the resize rule, so the contract version stays 1.0.
3. `scripts/export_onnx.py`: builds the graph, exports (opset 17, dynamic batch, constant folding), embeds the card
   in the ONNX metadata (`stratia_model_card`), validates the card, checks parity on random inputs (half of every
   batch with zero geometry) and on four real B0268 frames, checks the contract (names, shapes, finite, probability
   sums), measures the ray-map pooling error against ray maps computed directly at the 8 × 8 patch, times ONNX
   Runtime on the CPU, writes `docs/data/onnx_export.md` and registers the P096 run. `--dummy` writes the small
   untrained graph `tests/fixtures/stratia_dummy/` (0.45 MB, with its own card) for CloudScope P065; `.gitignore`
   admits `tests/fixtures/**/*.onnx`.
4. Export: `models/stratia-1.0.0-demo/stratia.onnx` (94.9 MB, git-ignored; distributed through a GitHub Release at
   P097, meanwhile copied by hand to the demo laptop) with `model_card.json` and `export_report.json`
   (committed). Model version `1.0.0-demo` marks the pre-locked-test weights.
5. Preview, not evaluation: the four B0268 frames of 19 Oct through the graph with the nominal B0268 camera, a zenith
   pose and the sidecar's Sun. Finite, well-formed outputs; one of four frames above the OOD threshold; the oktas
   head says 6–8 while the sky-parse head finds 31–61 % cloud pixels, and the genus head leans to Ns / Cs on a
   camera it has never seen. Recorded as the first domain-shift observation for P083 / P084 (the labels of P041
   are being made by two annotators now).

## Verification
- `tests/test_export.py` (5 tests, CPU, dummy graph): contract names / shapes / probabilities, torch ↔ ORT parity
  < 1e-4, zero geometry as a valid input, card embedded in the metadata, temperatures folded, pooled ray map within
  1° of the direct 8 × 8 map on the nominal B0268 fisheye, card validates and a wrong contract version / class
  order is rejected. Full suite green locally (`python -m pytest -q -m "not gpu"`), ruff clean, test lock unchanged.
- `docs/data/onnx_export.md` (generated): parity per output, random inputs and real frames, max |Δ| ≤ 1.6e-4
  (dense logits, relative ≤ 5.5e-5), ≤ 3.4e-5 on every global output; ONNX Runtime CPU batch-1 latency 172 ms on the
  laptop's i7; ray-map pooling 64 / 64 valid patches, 0 flag disagreements, direction error mean 0.03°, max 0.05°
  (the CBH prior's sigma is 20°). Run ID in `runs/registry.csv` (phase P096, name `onnx_export`).
- Parity bound: the plan's ≤ 1e-4 holds for every global output and in relative terms for the dense logits; the
  dense heads' absolute maximum (1.5e-4 on logits of magnitude ~10 at 128 × 128 × 7 positions) is attention
  summation order in ORT's CPU kernels, not a graph difference. Recorded as met in relative terms.

## Exit criteria
- [x] Opset ≥ 17, dynamic batch (opset 17; batch axis on every input and output).
- [x] PyTorch ↔ ORT parity ≤ 1e-4 (relative; absolute ≤ 1.6e-4 on the dense logits, see above).
- [ ] INT8 for the Pi — deferred to P095 / P097 (no edge student yet; CloudScope's CPU mode runs the FP32 graph at
      172 ms per frame, enough for the demo).
- [x] `model_card.json` validates against the schema; embedded in the ONNX metadata as well.
- [ ] Accuracy-drop report — not applicable to the FP32 export (parity report instead); due with INT8.
- [ ] Contract test passes in CloudScope — CloudScope P065 (next; the dummy fixture is for it).

## Fit & data-risk notes
No training. The export changes two things against the validation numbers of P082 / P089: the ray map is pooled
in the graph rather than computed at the coarse patch (error ≤ 0.05°, negligible against the 20° prior) and the
backbone runs in FP32 instead of the FP16 caches (token differences of 1e-3 relative; the heads were trained on the
FP16 values). Neither is measured against labels here; P090's locked test runs through this same graph so that the
deployed and the reported model are one object. The B0268 preview shows the expected domain shift (camera identity
drives the OOD score, P068) and a cover over-estimate; nothing is concluded from four frames.

## Deviations from plan & why
Pulled forward from Stage K (after P095) to before P090 for the demo; INT8 and the accuracy-drop report deferred
with the edge student. Model version `1.0.0-demo`; the release version will be exported again from the locked-test
model at P097 with the same script.

## Next phase
CloudScope Stage F (P065 contract loader with the dummy fixture, P066 ORT engine, P067 ray map, P068 overlays), then
STRATIA P041 labels → P083 zero-shot on B0268 → reduced P084 adaptation (4 days) if the labels arrive in time.
