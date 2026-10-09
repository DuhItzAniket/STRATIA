# P046 — Geometry-consistent transforms

Status: DONE     Date: 2026-10-10     Commit: (this commit)

## Objective
Make it impossible for a crop, resize, flip or rotation to leave the ray map behind: every geometric augmentation
is a pixel remapping tracked in one state, and the ray map is recomputed from the camera through that state.
Decision recorded as ADR-004 (native camera view + ray maps; no re-projection; augmentations as remappings).

## Inputs / dependencies
P043 camera models; P044 poses; P045 ray-map generator (`patch_rays`, `ray_map`, `full_frame_affine`); the P022
image cache (longest side 768); ADR-012 pixel-centre convention.

## Work log
1. `docs/adr/ADR-004-native-view-geometry.md`: the model consumes the native view; geometry travels through the
   pixel map, never as an image; the ray of an output patch is the ray of the source pixel its centre came from
   (flips carry directions, they do not mirror the world); padding is invalid; the cache is a transform.
2. `stratia/geometry/transforms.py`: `GeoState` (current size, 3 × 3 affine current → source pixels, coverage image,
   source size; `identity`, `cached`, `to_source`, `from_source`); transforms `Resize`, `Crop` (zero padding
   outside the box), `RandomResizedCrop` (torchvision-style box sampling), `HorizontalFlip`, `Rotate` (about the
   image centre, `cv2.getRotationMatrix2D` convention), `Compose`; `geometry_for(state, camera, pose, Sun, mask)`
   → (ray map through the composed map with padding patches invalid, camera mask in current pixels ∧ coverage);
   `patch_fraction`. Images and coverage are warped with the same matrices (linear / nearest).
3. Ray maps are never resampled: `ray_map` takes the affine and recomputes directions at the output patch centres.

## Verification
`tests/test_transforms.py` (8 tests):
- a coordinate-encoding image pushed through Resize → Crop → Rotate → Flip → Resize decodes, at every fully
  covered patch centre, the source coordinates the composed map predicts (max error < 1 px, the interpolation
  rounding); the inverse map round-trips;
- the cached-image state composed with the model resize equals the contract's full-frame affine (1e-9);
- the Sun stays at the Sun: over random chains of RandomResizedCrop, Rotate (±180°), HorizontalFlip (p = 0.5) and
  Resize, the patch holding the projected Sun looks within 6° of it with Sun-relative azimuth within 15° (a patch is
  ~6° wide at the image centre);
- a horizontal flip gives exactly the mirrored zenith-angle and azimuth maps; a 37° rotation gives the zenith map
  of the rotated dense zenith image within 1° at every fully covered patch;
- padding from a crop outside the frame is invalid in the ray map (zeros in all four channels) and false in the
  mask; RandomResizedCrop boxes stay inside the image; an unknown camera gives an all-zero ray map but a coverage
  mask.
Whole suite 149 passed; `ruff check .` clean.

## Exit criteria
- [x] Unit test: transform(image, ray) consistency.

## Fit & data-risk notes
- The guard the plan names ("augmentation that breaks geometry = silent label noise") is now structural: a
  transform that bypasses `GeoState` cannot produce a ray map at all. P048 (augmentation policy) must build its
  geometric part from these transforms; photometric ones leave the state alone.
- Flips produce a left-handed layout of directions (ADR-004 §3); nothing in the targets depends on handedness.
  If a future target did (e.g. a Sun-side/anti-Sun-side asymmetry learned from layout rather than from the ray
  map), flips would have to be dropped for it.
- Rotation about the image centre is a rotation about the zenith only up to the calibration-centre offset
  (Eye2Sky 20–50 px); the ray map is exact regardless.

## Deviations from plan & why
None. (P047 pairing and P048–P050 follow; P047 waits for the OLDLR/WESTE download, which is running.)

## Next phase
P047 — Image–ceilometer pairing (when the ceilometer-site images are on disk); P048 — Augmentation policy.
