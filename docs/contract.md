# STRATIA ↔ CloudScope interface contract — `stratia-contract` v1.0

Status: v1.0 (P008, 2026-10-04) · Implemented on the CloudScope side in CloudScope P065/P067 · Conventions: CloudScope ADR-012

STRATIA is distributed as an ONNX model plus a `model_card.json`. CloudScope (or any other host) supplies an image and optional geometry; STRATIA returns predictions with uncertainty. STRATIA has no state and controls nothing.

## 1. Conventions (shared with CloudScope ADR-012)

- Time in UTC; azimuth from geographic north, clockwise; elevation from the horizon (zenith = 90°); local frame East-North-Up.
- Image pixel (u, v) from the top-left pixel centre, u right, v down.
- Distances in metres.

## 2. Inputs

| Name | Type / shape | Required | Definition |
|---|---|---|---|
| `image` | float32 `[N, 3, H, W]` | yes | RGB, values scaled to [0, 1] then normalised with the model card's `mean` / `std`. `H`, `W` and the resize rule are in the model card (v1 default 512 × 512, resize of the full frame without cropping, camera-mask pixels set to 0 before normalisation). |
| `ray_map` | float32 `[N, 4, H/p, W/p]` (`p` = patch size, 16 in v1) | no (zeros allowed) | At each patch centre of the resized image: channels 0–2 = unit viewing direction **in a Sun-aligned local frame** (z up; x toward the Sun's azimuth; y = z × x); channel 3 = 1 if the direction is valid (calibrated, inside the camera mask, above the horizon), else 0. Zenith angle = acos(z); azimuth relative to the Sun = atan2(y, x). If the camera is uncalibrated or the Sun position is unknown, the whole tensor is 0. |
| `meta` | float32 `[N, 3]` | no (zeros allowed) | `[cos(Sun zenith), sin(Sun zenith), valid]`; all zero when time or location is unknown. Time of day, date and location are **deliberately not inputs** (they would let the model learn climatology shortcuts). |

Hosts compute `ray_map` from their own calibration and pose; STRATIA ships a reference implementation and shared test vectors (STRATIA P043–P045, CloudScope P031/P067).

## 3. Outputs

| Name | Shape | Meaning |
|---|---|---|
| `genus_logits` | `[N, 12]` | Multi-label logits (apply sigmoid): Ci, Cc, Cs, Ac, As, Ns, Sc, St, Cu, Cb, clear, contrail (order fixed by the model card's `genus_classes`). |
| `etage_logits` | `[N, 3]` | Multi-label logits: low, mid, high cloud present. |
| `oktas_probs` | `[N, 9]` | Probabilities for 0…8 oktas of total cloud cover (sum to 1). |
| `sky_parse_logits` | `[N, 4, H/4, W/4]` | Per-pixel classes: invalid/obstruction, sky, cloud, sun/glare. |
| `layer_logits` | `[N, 3, H/4, W/4]` | Per-pixel cloud layer: low, mid, high (meaningful on cloud pixels only). |
| `cbh_probs` | `[N, K+1]` | Probabilities over K cloud-base-height bins (edges in metres in the model card, ending at 12,000 m) plus a final "no cloud overhead" class. Refers to the lowest cloud base along the zenith direction; for non-zenith views it refers to the optical axis. |
| `ood_score` | `[N, 1]` | Higher = less trustworthy input. The model card gives `ood_threshold`; above it, hosts must show results as "uncertain". |

Calibrated probabilities: hosts apply the per-head temperatures in the model card (`calibration.temperatures`) to logits before sigmoid/softmax, unless the card states they are folded into the exported graph.

## 4. Model card (`model_card.json`)

Validated against [`schemas/model_card.schema.json`](../schemas/model_card.schema.json). Required content: contract version, model name and version, creation time and git SHA, input specification (size, resize rule, mean/std, patch size), class lists, CBH bin edges, calibration temperatures and OOD threshold, training-data manifest hashes, licences (including "Built with DINOv3" attribution), and the evaluation report reference.

## 5. Versioning

`MAJOR.MINOR`. A change to any input or output name, shape, channel meaning or class order is a **major** change and must land in both repositories (STRATIA phase doc + CloudScope phase doc) before use. Adding optional fields to the model card is minor. Hosts refuse models whose major version they do not support.
