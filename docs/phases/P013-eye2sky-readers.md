# P013 — Eye2Sky readers

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Read Eye2Sky calibration files, camera masks and image file names correctly, including the fisheye camera model that maps pixels to viewing directions.

## Inputs / dependencies
P012 inventory; `Eye2Sky/asi_meta/` (38 calibration files with content, 1 empty).

## Work log
1. `stratia/geometry/ocam.py` — Scaramuzza/OCamCalib model: `pixel_to_ray(u, v)` (u = column, v = row) returning unit rays in the STRATIA/CloudScope camera frame (+x toward +u, +y toward +v, +z along the optical axis, ADR-012), and `ray_to_pixel` via a dense monotonic table inverting θ(ρ) = atan(F(ρ)/ρ).
2. **Axis convention measured, not assumed.** The YAML centre `(xc, yc)` could be (column, row) or (row, column). Comparing it with the centroid of each camera mask: **34 of 38 calibrations match `xc` = row**, the original OCamCalib convention; the 4 others are near-ties (masks are cut by obstructions). Adopted `xc` = row; P044 verifies against the Sun's position.
3. `stratia/data/eye2sky.py` — `parse_image_name` (UTC time and exposure suffix from `<YYYYmmddHHMMSS>_<exp>.jpg`, station from the `ASI_<date>_<station>` folder), `load_calibration` (returns `None` for the empty OLUOL 2020 file), validity-window selection, mask resolution and loading.
4. **Mask file-name mismatch handled:** the YAML hint says `masks/<DATE>_<STATION>_mask.mat` but the files are `<STATION>_<DATE>_mask.{png,mat}`; the resolver tries the real pattern first. PNG and MATLAB masks were confirmed identical (`Mask.BW` == PNG for all 38).

## Verification
- Tests: 9 passing — round trip < 0.1 px, unit rays, centre = optical axis, frame handedness, horizon inside the image, file-name parsing, empty-file handling, and real calibrations + masks for **OLDLR, WESTE and AURIC** (skipped in CI where the data is absent).
- On 20,000 random valid mask pixels per station: round-trip error max ≈ 4 × 10⁻⁷ px (OLDLR 4.09e-7, WESTE 3.86e-7, AURIC 3.86e-7); maximum off-axis angle inside the masks 84.9–87.8°.

## Exit criteria
- [x] Unit tests on 3 stations; pixel/ray round-trip error < 0.1 px.

## Fit & data-risk notes
A wrong axis convention would silently corrupt every ray map and Sun projection used for cloud-base height; it is now measured here and re-checked against the Sun in P044.

## Deviations from plan & why
- The OCamCalib camera model, planned for P043, is implemented here because the round-trip exit criterion needs it; P043 adds the B0268 (OpenCV) model and the "unknown camera" placeholder on the same interface.
- External orientation (roll, pitch, yaw) is parsed but not yet interpreted; its rotation convention is fitted against Sun positions in P044.

## Next phase
P014 — Ceilometer reader.
