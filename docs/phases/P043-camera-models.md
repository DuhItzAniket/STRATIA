# P043 — Camera models

Status: DONE     Date: 2026-10-10     Commit: (this commit)

## Objective
One interface for pixel ↔ viewing direction over every camera STRATIA meets: OCamCalib (Eye2Sky), OpenCV fisheye
and pinhole (the B0268 through CloudScope's calibration file), and an explicit "unknown camera" placeholder; a
registry that resolves a manifest row to its model; shared test vectors that CloudScope must reproduce.

## Inputs / dependencies
P013 (`stratia/geometry/ocam.py`, Eye2Sky calibration readers); CloudScope P031 (camera-model file schema
`cloudscope.camera_model/1`, `core/resources/camera_model.schema.json`); CloudScope ADR-012 / `docs/contract.md`
conventions; the B0268 datasheet (4656 × 3496, 105° horizontal field of view).

## Work log
1. `stratia/geometry/cameras.py`: the `CameraModel` protocol (`kind`, `calibrated`, `pixel_to_ray`, `ray_to_pixel`,
   `width`, `height`); `OpenCVCamera` with OpenCV's pinhole (k1 k2 p1 p2 k3) and equidistant fisheye (k1..k4)
   projections written in NumPy (Newton inversion for the fisheye angle, fixed-point for the pinhole, as OpenCV
   does) so geometry has no OpenCV dependency; `UnknownCamera` (NaN rays, `calibrated = False`); the CloudScope file
   loader with field checks mirroring the schema (no jsonschema dependency); `nominal_fisheye` for datasheet models;
   the registry `configs/cameras.yaml` and `camera_for(camera_id, width, height, calib_id)`.
2. `OcamModel` gained `kind` / `calibrated` so the three models share the interface; its mapping is unchanged.
3. `schemas/camera_model.schema.json`: copy of CloudScope's schema (the shared contract file).
4. **B0268 is provisional.** No checkerboard calibration exists yet (CloudScope P031 owner item), so
   `configs/cameras/b0268_nominal.camera.json` is an equidistant model with zero distortion whose horizontal field
   of view is the datasheet's 105° (f = 2540.7 px); `configs/cameras.yaml` marks it `provisional: true`. The
   datasheet says the lens is strongly distorted, so this model is for plumbing and the P049 view synthesis only;
   no B0268 geometry number may be reported from it.
5. `scripts/make_camera_test_vectors.py` → `tests/vectors/camera_test_vectors.json`: pixel → ray and ray → pixel
   samples for the real OLDLR OCamCalib calibration (valid from 2022-04-22) and the synthetic OpenCV fisheye and
   pinhole models CloudScope's own tests use (3 cameras, 39 samples, conventions stated in the file). CloudScope
   copies the file (ADR-012's shared vector); `tests/test_cameras.py::test_shared_test_vectors` recomputes it here.
6. `docs/contract.md` now names the reference implementation and the vectors file.

## Verification
- `tests/test_cameras.py` (13 tests): interface and round trip on all three models (max error < 0.01 px over a
  25 × 25 grid), NumPy projections against `cv2.fisheye.projectPoints` / `cv2.projectPoints` (1e-3 px) and
  `undistortPoints` with tight iteration criteria (1e-6), ADR-012 axes, directions behind the camera refused,
  unknown camera, nominal field of view (52.5° at the image edge within 0.05°), schema validation cases, registry
  resolution for every manifest camera type, the shared vectors.
- The pinhole with tangential terms bends an on-axis-row pixel 4e-5 off the axis: expected, the test allows 1e-3.
- Whole suite 149 passed; `ruff check .` clean.

## Exit criteria
- [x] Round-trip tests pass (OCam, OpenCV fisheye, OpenCV pinhole; < 0.01 px).

## Fit & data-risk notes
- A provisional camera model is a silent-geometry risk if forgotten: the registry flag exists so the loader and
  the model card can refuse to report B0268 geometry results until the calibrated file replaces it.
- OpenCV's models stop at 90° off-axis; the B0268's 105° field is inside that, the Eye2Sky fisheyes (up to 88° in
  the masks) use OCamCalib, which is not limited.
- Uncalibrated datasets (CCSN, MGCD, Montenegro, SWIM family, Almería) get the unknown camera and therefore
  all-zero ray maps, which is what the contract specifies and what geometry dropout (P050) trains for.

## Deviations from plan & why
- The OCamCalib model was implemented in P013; this phase adds the OpenCV models, the placeholder, the registry and
  the vectors.
- The B0268 model is nominal, not calibrated (owner item: print a checkerboard, run CloudScope's camtool).

## Next phase
P044 — Sun position & calibration validation.
