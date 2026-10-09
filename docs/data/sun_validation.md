# Sun-based calibration validation (P044)

Generated 2026-10-09T20:22:36+00:00 by `scripts/sun_validation.py` (sampling: every 7th day, every 5 min, Sun >= 5.0 deg).

## Method

The Sun disc is detected as the one saturated, disc-like blob inside the camera mask; its pixel becomes a camera ray through the station's OCamCalib model (P013), pvlib's SPA gives the Sun's apparent East-North-Up direction at the frame time, and a rotation camera -> ENU is fitted by Kabsch with outlier trimming. The declared `external_orientation` [roll, pitch, yaw] of the calibration files is read under every plausible convention (768 candidates: six Euler orders x eight sign patterns x four base frames x two camera frames x camera-to-world or world-to-camera) and the reading with the lowest mean median error wins.

## Fitted poses

| Calibration | Station | Days | Sampled | Detected | Inliers | Median err | p90 err | Declared reading err | Fitted vs declared | Axis elevation | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AURIC_20191209.yaml | AURIC | 2022-04-01 to 2022-07-29 | 3084 | 129 | 80 | 0.079 deg | 0.258 deg | 0.347 deg | 0.359 deg | 89.57 deg | **ok** |
| BARSE_20191025.yaml | BARSE | 2022-04-01 to 2022-07-29 | 3080 | 83 | 40 | 0.120 deg | 0.363 deg | 0.156 deg | 0.089 deg | 88.71 deg | **ok** |
| OLDLR_20220131.yaml | OLDLR | 2022-04-01 to 2022-04-08 | 718 | 61 | 51 | 0.096 deg | 0.247 deg | 0.102 deg | 0.119 deg | 88.81 deg | **ok** |
| WESTE_20220307.yaml | WESTE | 2022-04-01 to 2022-04-08 | 717 | 41 | 30 | 0.115 deg | 0.362 deg | 2.943 deg | 3.035 deg | 87.36 deg | **ok** |

Detection rate over all sampled frames: 4.1% (314 of 7,599); undetected frames by reason: {'no disc-like prominent blob': 2836, 'no pixel above threshold': 2700, 'several candidate blobs': 876, 'brightest region is not the blob': 873}.

Status rule: at least 15 inliers spanning >= 3.0 h, median <= 1.0 deg, p90 <= 2.0 deg.

## Reading of the declared orientation

Winner: `yxz|-r+p+y|ENU|ocam|c2w` (mean median error 0.887 deg); runner-up `zxy|+r-p-y|ENU|ocam|w2c` at 0.887 deg.

| Candidate | AURIC_20191209.yaml | BARSE_20191025.yaml | OLDLR_20220131.yaml | WESTE_20220307.yaml | Mean |
|---|---|---|---|---|---|
| `yxz|-r+p+y|ENU|ocam|c2w` | 0.347 | 0.156 | 0.102 | 2.943 | 0.887 |
| `zxy|+r-p-y|ENU|ocam|w2c` | 0.347 | 0.156 | 0.102 | 2.943 | 0.887 |
| `xyz|+r+p+y|ENU|ocam|c2w` | 0.346 | 0.154 | 0.108 | 2.942 | 0.887 |
| `zyx|-r-p-y|ENU|ocam|w2c` | 0.346 | 0.154 | 0.108 | 2.942 | 0.887 |
| `xzy|+r-p-y|ENU|ocam|c2w` | 0.310 | 0.335 | 0.715 | 2.620 | 0.995 |
| `yzx|-r+p+y|ENU|ocam|w2c` | 0.310 | 0.335 | 0.715 | 2.620 | 0.995 |
| `xyz|+r-p+y|ENU|ocam|c2w` | 0.346 | 0.373 | 0.972 | 3.006 | 1.174 |
| `zyx|-r+p-y|ENU|ocam|w2c` | 0.346 | 0.373 | 0.972 | 3.006 | 1.174 |

Name: `<Euler order>|<signs of roll, pitch, yaw>|<base frame>|<camera frame>|<direction>`; the Euler rotation `R_order[0] R_order[1] R_order[2]` is built in the base frame, acts on the named camera frame (ADR-012 standard or OCamCalib's x-along-rows, y-along-columns, z away from the scene) and is read camera-to-world (`c2w`) or world-to-camera (`w2c`).

## Manifest consistency

`sun_zenith_deg` / `sun_azimuth_deg` versus `stratia.geometry.sun.sun_position`: 300 Eye2Sky manifest rows: max |difference| 0.00e+00 deg in zenith and azimuth.

## Figures

- `docs/data/figures/sun_validation_AURIC.jpg`
- `docs/data/figures/sun_validation_BARSE.jpg`
- `docs/data/figures/sun_validation_OLDLR.jpg`
- `docs/data/figures/sun_validation_WESTE.jpg`

## Decisions

- Ray maps (P045) use the **fitted** pose where one has status `ok` and the declared orientation under the winning reading elsewhere (`stratia.geometry.pose.pose_for`); the table above is the expected accuracy of each.
- A calibration whose fitted pose is `excluded` is not used for cloud-base-height pairing (P047) until it is understood.
- Stations without images on disk (all except those above) carry the declared reading; its accuracy on the tested stations is the column *Declared reading err*. Rerun this script when images of a new station arrive (OLDLR).
