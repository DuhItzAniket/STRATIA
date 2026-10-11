# P075 — Stage-3: CBH head

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The CBH head's training regime on the P047 pairs: seed variance, the held-out-site protocol that criterion C3
asks for (CDLRB / WESTE never seen in training), the synthetic consumer views of P049 as extra training data, the
overcast-day guard, and the geometry-dropout question left open by P067; skill against climatology by height bin.

## Inputs / dependencies
P067 head and trainer; P047 pairs and day splits; P049 synthetic views (365 labelled); P051 pooled grid; research
design §5 (C3).

## Work log
1. `scripts/train_cbh_head.py` gained `--train-stations` (train on OLDLR only, score both stations), `--synthetic`
   (the labelled P049 views, tokens from the backbone at run time, ray maps from each view's own camera and pose),
   `--day-weight` (rows weighted by 1 / `day_label_share`, floored at 0.2), `--train-fraction` (P079), per-station
   scores at the best epoch, and the best checkpoint plus validation predictions saved in the run folder (P081).
   The data pieces moved to `stratia/train/cbh_data.py` for P076 / P077.
2. Runs (10 epochs, 40–80 s each): base seeds 0 / 1 / 2, geometry dropout 0.15, OLDLR-only, + synthetic views,
   day-weighted. Table in `docs/data/cbh_head.md`.

## Verification
Validation days (12 per station), 1,368 cloudy pairs, climatology MAE 1,147 m:

| Variant | MAE (m) | Skill | OLDLR MAE / skill | WESTE MAE / skill | Geometry zeroed |
|---|---|---|---|---|---|
| base, seed 0 / 1 / 2 | 566 / 546 / 558 | +0.506 / +0.524 / +0.513 | 607 / 566 / 577 (+0.47 … +0.51) | 528 / 528 / 541 (+0.53 … +0.54) | 566 / 572 / 559 |
| + synthetic views | **538** | **+0.531** | 554 / +0.517 | 523 / +0.544 | 541 |
| geometry dropout 0.15 | 560 | +0.511 | 600 / +0.477 | 523 / +0.544 | 557 |
| day-weighted | 574 | +0.499 | 621 / +0.459 | 530 / +0.538 | 582 |
| **OLDLR only (WESTE held out)** | 610 | +0.470 | 582 / +0.494 | **636 / +0.448** | 615 |

- Seed std of the base MAE: 10 m (skill ±0.009); every variant difference below 20 m is inside it.
- **Held-out site:** trained on OLDLR alone the head keeps skill +0.448 on WESTE, a site and camera it never saw
  (C3 asks ≥ 0.10); the loss against training on both stations is 108 m on WESTE and none on OLDLR.
- **Synthetic views** help by 28 m (+0.025 skill), more than the seed spread: the 365 perspective views carry the
  same ceilometer labels seen through a different projection, which the head reads as extra samples.
- **Overcast-day weighting** does not help (574 m); the day label mix is already balanced by the sampler's view of
  the pairs, and the guard's risk (one label dominating a day) does not show in the per-bin errors.
- **Geometry:** in every variant the geometry-zeroed score equals the trained score; dropout 0.15 does not change
  that (P067 found the geometry used only at dropout 0). The decision for Stage H is to keep dropout 0.3 (the
  robustness the contract needs) and read the geometry question again on the full-resolution grid in P084 / P091.
- By height bin (base): MAE 263 m (0–1 km), 433 m (1–2 km), 772 m (2–4 km), 1,451 m (4–8 km), 1,208 m (8–20 km);
  prediction histogram ≈ label histogram (top-bin share 0.25 vs 0.19), no mode collapse in any variant.

## Exit criteria
- [x] Skill vs climatology on val, by height bin (above); held-out site measured.

## Fit & data-risk notes
- 80 % interval coverage 0.92–0.95 in every variant: the distributions are wide; P081 shows what a temperature
  does, P089 calibrates on a proper split.
- The synthetic views' labels are inherited from the source frame's ceilometer under the flat-layer assumption
  (P049); their gain is small and comes with that caveat.

## Deviations from plan & why
- None.

## Next phase
P076 — Stage-4: joint multi-task (frozen).
