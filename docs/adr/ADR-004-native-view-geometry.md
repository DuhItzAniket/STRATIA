# ADR-004 — Native camera view plus ray maps; augmentations are pixel remappings

Status: Accepted     Date: 2026-10-10     Phase: P046

## Context
STRATIA must serve fisheye all-sky imagers (Eye2Sky, MGCD), a 105° consumer camera on a pan-tilt head (the owner's
B0268) and uncalibrated photographs (CCSN) with one model. Two ways to make the geometry comparable were open:
(a) re-project every image into a canonical view (an equiangular or perspective sky map) before the network sees
it, or (b) feed the network the image as the camera produced it and tell it, per patch, where each patch looks
(the contract's `ray_map`, docs/contract.md). The second question is what happens to that geometry under
training-time augmentation: a crop, resize, flip or rotation changes which sky direction a patch shows, and a ray
map that is not transformed with the image is silent label noise on every geometry-dependent target (cloud-base
height at the zenith, the Sun-relative glare class, layer heights).

## Decision
1. **Native view.** The model consumes the image in the camera's own projection, resized to the contract's input
   size without cropping at inference. No re-projection to a canonical sky map: uncalibrated images have no
   projection to re-project from, re-projection resamples texture that the cloud classifier depends on, and the
   contract's ray map already carries the geometry continuously (unit vectors in a Sun-aligned frame + valid flag).
   Consumer-view synthesis (P049) is the one place where images are re-projected, as a training-data generator.
2. **Geometry travels through the pixel map, never as an image.** A `GeoState` (stratia/geometry/transforms.py)
   holds the affine map from current pixels to source pixels and a coverage image. Every geometric transform
   warps the image, warps the coverage and composes the map; the ray map is computed afterwards from the camera
   model, the pose and the Sun through the composed map (`geometry_for`). Ray maps are never resized, rotated or
   flipped as arrays.
3. **The ray of an output patch is the ray of the source pixel its centre came from.** This holds for flips and
   rotations too: after a horizontal flip the content at a patch still came from a definite sky direction and the
   ray map says which. The world frame is not mirrored to compensate; the consequence is that a flipped sample has
   a left-handed layout of directions, which no target in this project depends on (cloud appearance per direction
   is handedness-free). Padding (content outside the source after a crop or rotation) is invalid in the ray map and
   zero in the image mask.
4. **The cache is a transform.** The P022 image cache (longest side 768) is a plain resize of the source; the loader
   starts its `GeoState` from the cached size and the manifest's source size, so the geometry refers to the
   original calibration regardless of what resolution the pixels are read at.
5. Photometric augmentations (P048) do not touch the state.

## Consequences
- Augmentation code cannot forget the geometry: a transform that does not go through `GeoState` cannot produce a
  ray map at all. The unit tests (tests/test_transforms.py) check that a coordinate-encoding image and the composed
  map agree, that the Sun's patch still points at the Sun after any chain, that a flipped zenith map is the mirror
  of the original and a rotated one the rotation of the original, and that padding is invalid.
- Rotation about the image centre is a rotation about the optical axis only when the calibration centre is the
  image centre (Eye2Sky: 20-50 px off). The ray map is exact either way; the augmentation simply samples a slightly
  different set of directions.
- Hosts (CloudScope) implement only the inference case, the full-frame resize, which is the identity transform of
  this scheme; the shared test vectors cover it.
