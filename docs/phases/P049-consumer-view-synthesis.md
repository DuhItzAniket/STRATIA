# P049 — Consumer-view synthesis

Status: DONE     Date: 2026-10-10     Commit: (this commit)

## Objective
Bridge the all-sky → consumer-camera gap with data: re-project Eye2Sky fisheye frames into 105° perspective views at
random tilt and azimuth, as the owner's B0268 on its pan-tilt head would see them, with exact ray maps and, at the
ceilometer sites, inherited cloud-base labels.

## Inputs / dependencies
P043 (nominal B0268 lens, OCamCalib models), P044 (fitted poses: the view's directions are only as good as the
source pose, 0.08–0.12°), P045 (ray maps), P036 (ceilometer target table), Eye2Sky frames on disk (AURIC and BARSE
April–July, OLDLR and WESTE 1–12 April so far; the download is running).

## Work log
1. `stratia/geometry/synthesis.py`: `Pointing` and `pointing_pose` (optical axis at azimuth/elevation, image-up
   toward the sky, optional roll); `random_pointing` (elevation 15–70°, any azimuth, roll ±3°); `scaled_camera`
   (the nominal B0268 at 1024 × 769 px, same field of view); `render_view` (virtual pixel → ray → ENU → source
   camera → source pixel, `cv2.remap`; valid = inside the source, inside its mask, above the horizon; per-camera
   ray cache); `view_ray_map` (the P045 generator on the virtual camera and pose with the rendered validity as
   the mask); `source_footprint` (for figures); `inherit_label` (nearest 30 s grid point of the P036 table within
   15 s, valid windows only; returns the ceilometer label, base height, confidence, layers and the assumption).
2. `scripts/synthesize_views.py`: six frames per day (Sun ≥ 10°), three views each, views with < 50 % valid pixels
   discarded, JPEGs under `<cache_root>/synthetic_views/<station>/`, index `data/synthetic_views.parquet`
   (pointing, Sun, validity, label), report `docs/data/synthetic_views.md`, figure
   `docs/data/figures/synthetic_views.jpg`. Stations can be run separately (the index merges).
3. **Label rule (provisional).** The ceilometer measures the base *height* at the zenith; a pointed view inherits
   that height under the flat-layer assumption (horizontally uniform layer over the few kilometres the view
   spans), not a slant range. Every inherited label carries `label_assumption = "flat layer"`; P047 fixes the
   pairing rule (zenith ROI, subsampling) and this hook adopts it.

## Verification
- `tests/test_synthesis.py` (6): pointing axes (facing north: image right = east, image down = toward the ground,
  roll keeps the axis), scaled camera keeps the field of view (0.1°), **direction fidelity**: a source frame whose
  colour encodes the ENU direction of every pixel is re-projected and decoded back at every valid view pixel,
  median error < 1.5° (8-bit colour quantisation), the view ray map's centre patch looks at the Sun placed on the
  axis, the footprint of a high view stays inside the source, label inheritance picks the nearest valid grid point
  and refuses invalid or distant ones.
- Generated set (`docs/data/synthetic_views.md`):

| Station | Views | Days | With label | Label mix | Mean valid |
|---|---|---|---|---|---|
| AURIC | 320 | 18 | 0 | — | 82.5 % |
| BARSE | 316 | 18 | 0 | — | 80.9 % |
| OLDLR | 211 | 12 | 167 | low 77, mid 42, none 21, high 18, mixed 9 | 83.0 % |
| WESTE | 216 | 12 | 198 | low 84, mid 45, none 39, high 24, mixed 6 | 83.3 % |

  1,063 views, 365 labelled (mean confidence 0.97, grid gap 0 s: frames sit on the 30 s grid). Rendering: ~1 s per
  frame including the three views.
- Visual check (figure): footprints where the views look, the horizon with trees and poles at the bottom of the
  views as a pointed camera shows it, zenith contours concentric around the zenith direction, the Sun where the
  ray map says.
- Whole suite passes; `ruff check .` clean.

## Exit criteria
- [x] Synthetic set + visual check.

## Fit & data-risk notes
- The views keep the source exposure (Eye2Sky's 0.15 ms day exposure, dim): a real B0268 auto-exposes. The P048
  exposure augmentation and per-sample normalisation cover the brightness gap; P075 may add a fixed gain.
- Nominal lens: the views carry the datasheet's 105° equidistant geometry, not the real B0268 distortion (P043,
  owner's checkerboard pending). The ray maps are exact for the lens used, so training stays self-consistent; the
  domain gap to the real lens is what P083 measures.
- Labels under the flat-layer assumption are wrong on broken or sloping layers; the ceilometer's `mixed` and
  `layers` columns travel with the label so P075 can weight or exclude them. These views supplement, never
  replace, real pairs (P047).
- Day blocks: synthetic views inherit the source frame's day for the splits (P037); they must never cross a split
  their source day does not.

## Deviations from plan & why
- The plan's "inherited CBH labels" are delivered through a provisional pairing rule (nearest grid point) that
  P047 will replace; the hook is one function.
- Set size is modest (1,063) on purpose: the generator is fast (~1 s per source frame) and P075 regenerates with
  its own sampling once P047's pairs and the full OLDLR/WESTE download exist.

## Next phase
P050 — Metadata dropout; then P047 — Image–ceilometer pairing.
