# P045 — Ray-map generator

Status: DONE     Date: 2026-10-10     Commit: (this commit)

## Objective
Produce the contract's geometry input for any sample: per patch the unit viewing direction in the Sun-aligned
frame and a valid flag, plus the Sun metadata vector, from a camera model, a pose, the Sun position, the camera
mask and a near-horizon floor; verify it visually on real frames.

## Inputs / dependencies
P043 cameras, P044 poses and Sun; `docs/contract.md` (`ray_map` float32 [4, H/p, W/p], p = 16, 512 px input,
full-frame resize; `meta` = [cos zenith, sin zenith, valid]); Eye2Sky masks (P013); `stratia/contract.py`.

## Work log
1. `stratia/geometry/raymap.py`: `full_frame_affine` (output pixel → source pixel for the contract's resize,
   pixel-centre convention); `patch_centres`; `patch_rays` (camera rays at the patch centres through an affine,
   `inside` flag, mask fraction from a 4 × 4 lattice per patch footprint); `ray_map` (Sun-aligned directions via
   the pose and `sun_aligned_frame`; valid = inside ∧ mask fraction ≥ 0.5 ∧ elevation ≥ floor; all-zero tensor for
   an unknown camera, a missing pose or an unknown Sun; invalid patches zero in every channel); `meta_vector`;
   `resized_mask` (for zeroing masked pixels before normalisation); derived views `zenith_angle_deg`,
   `azimuth_from_sun_deg`, `sun_distance_deg`; `near_horizon_table` / `min_elevation_for`.
2. `configs/near_horizon.csv`: Eye2Sky 5° (the Q25 compresses ~1° per 2 px there, the masks already stop at
   2–5° elevation, and P044 verified the poses at ≥ 5° only); B0268 0° (pointed camera, obstructions left to the
   sky-parsing head).
3. The affine parameter is what P046 builds on: the same generator serves the plain resize and every augmentation.
4. `scripts/ray_map_overlays.py`: for each station with images and a pose, the frame nearest 1 April 2022 12:00
   UTC → zenith-angle contours, the direction-toward-the-Sun field (arrow per second patch, coloured by
   |azimuth from Sun|), the valid flag, the projected Sun; `docs/data/ray_maps.md` with the Sun-patch check.

## Verification
- Tests `tests/test_raymap.py` (11): affine pixel-centre mapping, patch-centre grid, contract shape/dtype/unit
  norms/binary flag, the image-centre patch at the zenith (z > 0.995), the Sun's own patch within 4° of the Sun with
  Sun-relative azimuth within 10°, zero tensors for every unknown-geometry case, mask and horizon removing patches,
  mask fraction as a footprint share, resized mask centred, a pointed consumer camera (centre patch at the axis,
  top rows higher, bottom rows below the horizon), an OpenCV camera at a non-square size, the near-horizon table.
- Overlays (`docs/data/figures/raymap_<station>.jpg`, 1 April 2022 12:00 UTC, fitted poses):

| Station | Valid patches | Zenith range | Sun patch: distance to the Sun, azimuth from Sun |
|---|---|---|---|
| AURIC | 53.9 % | 0.45–82.5° | 1.71°, 0.0° |
| BARSE | 54.0 % | 1.0–84.3° | 0.72°, 0.6° |
| OLDLR | 52.3 % | 2.4–84.9° | 1.40°, −1.8° |
| WESTE | 56.1 % | 2.6–83.6° | 3.49°, −2.4° |

  The projected Sun sits on the visible disc in all four images; a patch is ~6° wide at the image centre, so the
  Sun-patch distances are the patch quantisation, not pose error (P044: 0.08–0.13°).
- Whole suite passes; `ruff check .` clean.

## Exit criteria
- [x] Visual check overlays (four stations), with the Sun-patch check as the numeric companion.

## Fit & data-risk notes
- Uncalibrated datasets get all-zero ray maps (the contract's "unknown" state), which is also the training-time
  dropout state (P050): the model must not learn to read "zeros" as a dataset identity. P029 showed dataset
  identity is readable from pixels anyway; the ray map must not add a second, trivial channel for it, so P050's
  dropout rate applies to calibrated datasets too.
- Mask fraction at 0.5 per patch: a patch half inside the mask is valid and its direction is that of its centre;
  the sky-parsing head sees the masked pixels as zeros (invalid class), so no cloud label leaks from outside the
  mask.
- The 5° floor removes ~2 % of Eye2Sky patches; cloud-base height uses the zenith region only.

## Deviations from plan & why
- The plan lists "camera mask + `near_horizon.csv`"; the CSV holds per-camera floors (patterns allowed) rather
  than per-station horizon profiles, which the masks already encode.

## Next phase
P046 — Geometry-consistent transforms.
