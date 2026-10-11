# P067 — CBH head

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
The cloud-base-height head: zenith-prior attention over the geometry-fused tokens, a distribution over the 24
étage-aligned height bins plus "no cloud overhead", trained with the ordinal CE + CRPS loss on the P047 pairs; it
has to beat climatology on the validation days and must not collapse onto the common bin.

## Inputs / dependencies
P047 pairs (10-minute cadence, train 81 / val 12 days per station); P051 cache + the new 8 × 8 pooled token grid
for the 20,942 CBH frames (`dinov3_vits16_512_ffr1-patches-cbh-pool4`, 1.0 GB); P045 ray maps at patch 64 from
the station calibration, the P044 fitted pose and the Sun; P062 `GeometryEncoder`; P064 `cbh_loss`; P058 baselines.

## Work log
1. `stratia/model/cbh.py`: `CbhHead(d_model, hidden=256, sigma_deg=20)`: attention logit = linear(LayerNorm(token))
   − (θ/σ)² with θ the zenith angle from the ray map's z channel; invalid patches get the horizon's prior, so a null
   ray map leaves a constant prior (content attention alone); pooled ⊕ CLS → MLP → 25 logits → softmax.
2. `stratia/features/cache.py`: `build(..., pool=k)` average-pools the stored patch grid (32 × 32 → 8 × 8 at
   k = 4); `scripts/build_feature_cache.py --patches cbh --pool 4` built the CBH grid in 7 minutes.
3. `scripts/train_cbh_head.py`: ray maps and meta per pair (one `patch_rays` per calibration, `ray_map` per frame
   with pvlib's Sun), geometry dropout at batch time, AdamW + one-cycle, 10 epochs of 40 s; scores on the val days
   (MAE / bias / skill of the median height against the train-median climatology, CRPS, 80 % interval coverage,
   cloud-presence balanced accuracy, the prediction-histogram guard) for the model as trained and for the same
   weights with the ray map and meta zeroed. `data/cbh_head.parquet`, `docs/data/cbh_head.md`.

## Verification
- `tests/test_cbh_head.py` (2): contract shape and normalisation; with zero-content tokens the attention peaks on
  the zenith patch of the OLDLR fisheye and is uniform for a null ray map; a toy training run moves the medians to
  their targets, keeps "no cloud" for a NaN target and trains σ.
- Validation days, 1,368 cloudy pairs (climatology MAE 1,147 m):

| Variant | MAE (m) | Bias | Skill | ≤ 500 m | CRPS | 80 % coverage | Cloud bal. acc | Top-bin share pred / true |
|---|---|---|---|---|---|---|---|---|
| head, geometry dropout 0.3 | **566** | +125 | **+0.506** | 0.69 | 0.0273 | 0.92 | 0.926 | 0.25 / 0.19 |
| same weights, geometry zeroed | 566 | +125 | +0.507 | 0.70 | 0.0273 | 0.94 | 0.921 | 0.26 / 0.19 |
| head, no geometry dropout | **551** | +80 | **+0.520** | 0.70 | 0.0269 | 0.94 | 0.924 | 0.26 / 0.19 |
| same weights, geometry zeroed | 765 | −528 | +0.333 | 0.62 | 0.0386 | 0.91 | 0.878 | 0.39 / 0.19 |
| P058 ridge on frozen DINOv3 | 699 | +34 | +0.390 | 0.61 | — | — | 0.909 | — |
| P058 ResNet-18 ceilometer-supervised | 724 | −42 | +0.368 | 0.63 | — | — | — | — |

- Reading: the head beats every P058 baseline (MAE 566 vs 699 m; skill +0.51 vs +0.39) on the same pairs, with
  the prediction histogram close to the label histogram (top-bin share 0.25 vs 0.19, entropy ratio 0.95: no mode
  collapse). **The geometry finding:** trained with dropout 0.3 the head learns a content-only solution (zeroing
  the geometry changes nothing; α = 0.12, σ unchanged at 19.6°); trained without dropout it uses the geometry
  (551 m, and 765 m when it is taken away). On this grid the geometry is worth ≈ 15 m of MAE when it is always
  present and nothing when it is sometimes absent: the dropout rate is a real trade between robustness to
  uncalibrated hosts and the use of calibration, to be set in P075 / P091, not a free regulariser.
- 80 % interval coverage is 0.92: the distributions are too wide (the soft target's σ = 0.15 and the CRPS term
  both reward width); temperature calibration (P089) will narrow them; the ranking is unaffected.

## Exit criteria
- [x] Trains on paired data; beats climatology on val (skill +0.51; also beats the supervised CNN by 22 % MAE).
- [x] Mode-collapse guard recorded (prediction histogram vs label histogram).

## Fit & data-risk notes
- Val is 12 days per station: the ±50 m differences between epochs 4–10 (572–619 m) are within day-to-day noise;
  the best epoch is chosen on val, as in P058, and the test days stay locked.
- The 8 × 8 grid makes the zenith cell 64 px wide (≈ 12° of sky on the fisheye); Stage H can rebuild at 16 × 16
  (4 GB) if P075's ablation says the geometry needs it.

## Deviations from plan & why
- The layer-consistency term is wired in the full model (P071 `total_loss` path) and not used in this standalone
  run, which has no étage head; P076 (joint training) is where it acts.

## Next phase
P068 — OOD / quality score.
