# Sun-based calibration validation (P044)

Generated 2026-10-11T03:57:49+00:00 by `scripts/sun_validation.py` (sampling: every 7th day, every 5 min, Sun >= 5.0 deg).

## Method

The Sun disc is detected as the one saturated, disc-like blob inside the camera mask; its pixel becomes a camera ray through the station's OCamCalib model (P013), pvlib's SPA gives the Sun's apparent East-North-Up direction at the frame time, and a rotation camera -> ENU is fitted by Kabsch with outlier trimming. The declared `external_orientation` [roll, pitch, yaw] of the calibration files is read under every plausible convention (768 candidates: six Euler orders x eight sign patterns x four base frames x two camera frames x camera-to-world or world-to-camera) and the reading with the lowest mean median error wins.

## Fitted poses

| Calibration | Station | Days | Sampled | Detected | Inliers | Median err | p90 err | Declared reading err | Fitted vs declared | Axis elevation | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OLDLR_20220131.yaml | OLDLR | 2022-04-01 to 2022-04-21 | 3125 | 110 | 70 | 0.435 deg | 0.912 deg | 0.486 deg | 0.258 deg | 88.92 deg | **ok** |
| OLDLR_20220422.yaml | OLDLR | 2022-04-22 to 2022-07-31 | 17811 | 224 | 134 | 0.882 deg | 1.905 deg | 9.345 deg | 10.878 deg | 88.87 deg | **ok** |
| WESTE_20220307.yaml | WESTE | 2022-04-01 to 2022-06-26 | 14539 | 236 | 155 | 0.677 deg | 1.353 deg | 2.547 deg | 3.427 deg | 86.93 deg | **ok** |
| WESTE_20220627.yaml | WESTE | 2022-06-29 to 2022-07-31 | 6263 | 92 | 44 | 0.940 deg | 1.797 deg | 0.924 deg | 0.463 deg | 86.69 deg | **ok** |
| AURIC_20191209.yaml | AURIC | 2022-04-01 to 2022-07-29 | 3084 | 129 | 80 | 0.079 deg | 0.258 deg | 0.347 deg | 0.359 deg | 89.57 deg | **ok** |
| BARSE_20191025.yaml | BARSE | 2022-04-01 to 2022-07-29 | 3080 | 83 | 40 | 0.120 deg | 0.363 deg | 0.156 deg | 0.089 deg | 88.71 deg | **ok** |

Detection rate over all sampled frames: 1.8% (874 of 47,902); undetected frames by reason: {'no disc-like prominent blob': 20552, 'no pixel above threshold': 11991, 'several candidate blobs': 7735, 'brightest region is not the blob': 6521, 'outside prior gate': 229}.

Status rule: at least 15 inliers spanning >= 3.0 h, median <= 1.0 deg, p90 <= 2.0 deg.

## Reading of the declared orientation

Winner: `yxz|-r+p+y|ENU|ocam|c2w` (mean median error 3.322 deg); runner-up `yzx|-r+p+y|ENU|ocam|w2c` at 3.322 deg.

| Candidate | OLDLR_20220131.yaml | OLDLR_20220422.yaml | WESTE_20220307.yaml | WESTE_20220627.yaml | Mean |
|---|---|---|---|---|---|
| `xzy|+r-p-y|ENU|ocam|c2w` | 0.710 | 9.379 | 2.277 | 0.919 | 3.322 |
| `yzx|-r+p+y|ENU|ocam|w2c` | 0.710 | 9.379 | 2.277 | 0.919 | 3.322 |
| `yxz|-r+p+y|ENU|ocam|c2w` | 0.486 | 9.345 | 2.547 | 0.924 | 3.326 |
| `zxy|+r-p-y|ENU|ocam|w2c` | 0.486 | 9.345 | 2.547 | 0.924 | 3.326 |
| `xyz|+r+p+y|ENU|ocam|c2w` | 0.485 | 9.355 | 2.545 | 0.929 | 3.329 |
| `zyx|-r-p-y|ENU|ocam|w2c` | 0.485 | 9.355 | 2.545 | 0.929 | 3.329 |
| `xzy|+r+p-y|ENU|ocam|c2w` | 0.910 | 9.337 | 2.777 | 0.918 | 3.485 |
| `yzx|-r-p+y|ENU|ocam|w2c` | 0.910 | 9.337 | 2.777 | 0.918 | 3.485 |

Name: `<Euler order>|<signs of roll, pitch, yaw>|<base frame>|<camera frame>|<direction>`; the Euler rotation `R_order[0] R_order[1] R_order[2]` is built in the base frame, acts on the named camera frame (ADR-012 standard or OCamCalib's x-along-rows, y-along-columns, z away from the scene) and is read camera-to-world (`c2w`) or world-to-camera (`w2c`).

## Manifest consistency

`sun_zenith_deg` / `sun_azimuth_deg` versus `stratia.geometry.sun.sun_position`: 300 Eye2Sky manifest rows: max |difference| 0.00e+00 deg in zenith and azimuth.

## Figures

- `docs/data/figures/sun_validation_OLDLR.jpg`
- `docs/data/figures/sun_validation_WESTE.jpg`

## Decisions

- Ray maps (P045) use the **fitted** pose where one has status `ok` and the declared orientation under the winning reading elsewhere (`stratia.geometry.pose.pose_for`); the table above is the expected accuracy of each.
- A calibration whose fitted pose is `excluded` is not used for cloud-base-height pairing (P047) until it is understood.
- Stations without images on disk (all except those above) carry the declared reading; its accuracy on the tested stations is the column *Declared reading err*. Rerun this script when images of a new station arrive (OLDLR).
