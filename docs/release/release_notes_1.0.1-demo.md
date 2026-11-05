# STRATIA 1.0.1-demo — release notes (prepared, not published)

Status: package built and hashed on 2026-10-23, release notes written 2026-11-05. **Nothing is published**: the
GitHub Release (or Hugging Face upload) is the owner's decision and action (P097 → P100).

## What it is

One ONNX graph (opset 17, FP32, dynamic batch; `stratia.onnx`, 94.9 MB, SHA-256 `66e22ed1…`, full digest in the
card) with its model card (`model_card.json`, also embedded in the ONNX metadata) and the export report. Inputs:
`image` [N, 3, 512, 512] (full-frame resize, no crop, ImageNet-normalised), `ray_map` [N, 4, 32, 32] (unit directions
in the Sun-aligned frame + valid flag; all zeros when unknown), `meta` [N, 3] (cos / sin of the Sun zenith, valid).
Outputs: calibrated genus logits (12), étage logits (3), oktas probabilities (9, CORAL), sky-parse logits
[N, 4, 128, 128], layer logits [N, 3, 128, 128], cloud-base distribution (24 bins + no cloud), OOD score. The P089
calibration (temperatures, conformal radii, OOD statistics) is folded into the graph; hosts apply the card's OOD
threshold (1.497) and conformal radii (80 %: 0.3265, 95 %: 0.4376 in CDF units) only.

## How it differs from 1.0.0-demo

The global heads were fine-tuned on the owner's camera (Arducam B0268, BLR01, three afternoons, 273 frames; P084,
heads only, 16× source replay, regression guard 0.02) and the calibration refitted (`configs/calibration_1.0.1.json`).
Dense and cloud-base heads, backbone and the contract are unchanged, so every public-dataset number of
`results/P090_locked_test.json` holds for both versions; the B0268 locked day (2026-10-20, 120 frames, read once)
is where they differ: étage-set agreement 0.000 → 0.463, oktas ±1 0.017 → 0.671, top genus in the raters' set
0.121 → 1.000, frames flagged OOD 0.508 → 0.000.

## Numbers (locked test, read once; `docs/data/locked_test.md`)

| Quantity | Value |
|---|---|
| LODO étage macro-F1 (mean of four folds) | 0.540 (probe 0.529) |
| in-domain genus macro-F1 (CCSN, MGCD) | 0.706 (ViT-L probe 0.733) |
| Montenegro étage-set / oktas ±1 agreement over the human ceiling | 0.926 / 0.963 |
| CBH skill OLDLR / held-out WESTE | 0.536 / 0.436; MAE 872 m (ridge 1,096 m) |
| 80 % / 95 % calibrated coverage | 0.720 / 0.918 |
| sky mIoU / layer mIoU | 0.900 / 0.668 |
| OOD AUROC (B0268 vs in-domain) | 0.939 |
| latency (P094) | 29 ms DirectML in CloudScope, 144 ms ONNX Runtime CPU, 20 ms PyTorch fp32 RTX 4050 |

## Known limitations

- Trained on one camera network and web / three-site images: the sky parse over-calls cloud on clear frames
  (0.28 of pixels on the clear-sky test set) and the glare class does not fire on the B0268 camera (circumsolar
  brightness is parsed as cloud); the conformal intervals fitted on spring weeks under-cover a summer season (0.72
  at the 80 % level). Zero-shot on a new camera is at the floor; adapt the heads (P084 recipe) first.
- The CloudScope application refines the masks it draws (image-guided upsampling to 512 px) and applies a
  geometric glare rule around the Sun when the geometry is known; those are presentation steps in the host, not
  part of this graph, and none of the numbers above include them.
- Licence: research use, non-commercial (see `docs/release/licences.md`).

## Package

`stratia-1.0.1-demo_2026-10-23.zip` (88.1 MB, 3 files: `stratia.onnx`, `model_card.json`, `export_report.json`),
SHA-256 `b41202aff36520706507bf1af50d68b25a76af84f4d8623a4cc252452a8ab4e4`; kept with
`B0268_2026-10-23.zip` (SHA-256 `cd7d2c3b…`) in the owner's OneDrive backup folder (`Backups/STRATIA/`,
`SHA256SUMS_2026-10-23.txt`). To publish: create a GitHub Release `v1.0.1-demo` on the STRATIA repository, attach
the zip and the SHA-256 line, paste this file as the body.

## Reproduce the tables from a fresh clone

```bash
python scripts/reproduce_tables.py
```

writes `paper/tables/*.{tex,md}` and `paper/numbers.tex` from `results/` alone (no data needed).
