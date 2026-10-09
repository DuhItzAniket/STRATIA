# P048 — Augmentation policy

Status: DONE     Date: 2026-10-10     Commit: (this commit)

## Objective
A training-time augmentation policy that models what sky cameras actually do to a frame (exposure, white balance
within a physical range, JPEG, noise, Sun glare, dirt and raindrops, obstructions) and what the geometry allows
(rotation about the zenith for all-sky imagers, flips and crops for pointed cameras), with dense labels and ray
maps following every geometric change, no hue shift anywhere, and the hooks the ablations need.

## Inputs / dependencies
P046 transforms and `geometry_for`; P045 ray maps; P044 poses (the Sun's pixel for glare); P018 dense labels
(`stratia.data.segmentation`, IGNORE = 255); the manifest's `camera_type`; P029 (fixed structure and Sun as
shortcuts); P030 (the all-sky / consumer / patch camera families).

## Work log
1. `stratia/geometry/transforms.py`: every transform takes an optional `label=` map and remaps it with nearest
   neighbour and IGNORE outside the source; `geometry_for(..., obstruction=)` treats a pasted obstruction like
   padding (patches mostly covered lose their valid flag; the mask output is False there).
2. `stratia/augment/photometric.py`: exposure (linear factor after gamma decoding, highlights clip), white balance
   (red/blue gains on the colour-temperature axis only, green fixed), JPEG, sensor noise (read + shot term in linear
   space), Sun glare (bloom at the Sun's pixel when the geometry is known, with a ghost mirrored through the image
   centre as the Q25 domes show, P044), dirt and raindrops, obstruction cut-outs (pole or branch-like silhouette;
   returns the mask).
3. `configs/augment.yaml` + `stratia/augment/policy.py`: camera-type groups (`all_sky`: fisheye_asi; `consumer`:
   wide_angle_usb, consumer_photo, fixed_lowcost; `patch`: wsi_crop, wsi_patch); geometric policy per group
   (all-sky: crop 0.8–1.0 of the area, rotation ±180°, no flip; consumer: crop 0.6–1.0, rotation ±5°, flip 0.5;
   patch: crop 0.7–1.0, rotation ±10°, flip 0.5); photometric probabilities and ranges; `strength` scales every
   range (P080) and every augmentation has `enabled` (P091). `AugmentationPolicy(image, camera_type, rng, state,
   label, sun_px_source, valid_source)` returns the image, the geometry state, the transformed label with
   obstructions set to the invalid class, the obstruction mask and the list of what fired.
4. `scripts/augment_gallery.py` → `docs/data/figures/augmentation_gallery.jpg`, `docs/data/augmentation_gallery.md`
   (policy table).

## Verification
- `tests/test_augment.py` (13): exposure monotonic with clipping; white balance leaves green untouched, moves red
  and blue oppositely and changes hue by < 6° on a sky; the policy file contains no hue/HSV/saturation entry;
  noise and JPEG change images by a few grey levels; glare brightens the Sun's pixel and respects the mask; dirt
  darkens; obstruction masks are plausible and the pasted pixels are dark; groups and chains per camera type;
  strength 0 reduces the policy to the plain resize; strength scales rotation ranges; with an obstruction the
  transformed label is 0 there, the ray-map patches lose validity and the mask is False; a ray map after the
  policy's geometric chain equals the direct one (1e-6); the policy runs for every group.
- Gallery reviewed (one Eye2Sky, CCSN, Montenegro and SWIMSEG image; each augmentation alone and three draws of the
  full policy): cloud texture survives every augmentation at strength 1; the bloom sits on the Sun for the Eye2Sky
  frame; the cyan valid-flag outline follows the mask, the crop and the pasted obstruction; SWIMSEG's cloud
  contours move with the crop and rotation. Owner's look at the gallery pending (exit criterion wording).
- Whole suite passes; `ruff check .` clean.

## Exit criteria
- [x] Visual gallery produced and reviewed (owner approval pending; nothing in the policy depends on it).
- [x] Per-augmentation ablation slot: `enabled` switches and `strength` in `configs/augment.yaml`.

## Fit & data-risk notes
- Too strong destroys cloud texture (underfit), too weak overfits (the plan's guard): strength is a single knob
  for P080, and the gallery at strength 1 keeps texture visible in every panel.
- White balance stays on the colour-temperature axis and no hue shift exists: sky colour carries cloud and
  aerosol information and must not be randomised away.
- Obstructions are the only augmentation that edits labels; they write the invalid class and clear ray-map
  validity, so the model never learns "cloud" or "sky" on a pasted silhouette.
- Glare is geometry-aware on calibrated frames and random elsewhere; the random case is a deliberate mismatch
  that teaches glare appearance without a Sun prior.

## Deviations from plan & why
- "Rotation about the zenith" is a rotation about the image centre (ADR-004 §: exact ray map either way).
- Dirt/raindrop and flare models are procedural, not measured; P080 may tune their frequencies.

## Next phase
P049 — Consumer-view synthesis.
