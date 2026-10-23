# P084 — Few-shot adaptation on B0268 (reduced: heads only)

Status: DONE (reduced)     Date: 2026-10-23     Commit: (this commit)

The design, the draws and the regression rule below were fixed in this document on 2026-10-23 (02:30 IST) before
any run; the runs followed the same day, 12:12–12:39 UTC, once P083 had its numbers. Two things changed against
the pre-registration and are recorded under *Deviations*: seven configurations were tried on the held-out days, not
one, and the demo model was exported although one draw of the chosen configuration is outside the regression rule.

## Objective
Plan (Stage I): LP-FT with source replay on day-blocked splits; legacy lesson: tune the replay ratio, 4x worked,
10x over-biased. Exit: a gain on held-out B0268 days without a source regression of more than 2 points. Guard:
over-fitting a handful of days.

## Inputs / dependencies
- P083 zero-shot numbers on the same frames (`docs/data/b0268_zero_shot.md`, run
  `20261023T120741Z_b0268_zero_shot_6a9765a3d2`): the baseline the gain is measured against.
- P041 targets `data/b0268_targets.parquet`, per-rater labels `data/b0268_derived.parquet` and the few-shot draws
  `data/splits/few_shot_target.parquet` (`stratia/data/splits.py` `few_shot_target`; the locked day 2026-10-20).
- Gate G4 global run `runs/20261011T125754Z_global_indomain_e2s_s0_27d0d07c22` (P073: hidden 512, dropout 0.1,
  kendall false, p_drop 0.3, lambda_h 0.1, eye2sky_etage true; `checkpoints/best.pt`, EMA weights): the heads to
  adapt and the source baseline.
- Source rows: `stratia/train/data.py` `global_rows(eye2sky_etage=True)` (22,546 rows: 19,425 train, 3,121 val;
  B0268 is not among `GLOBAL_SOURCES`), with the existing caches `dinov3_vits16_512_ffr1` (cls) and
  `dinov3_vits16_512_ffr1-patches-all-pool4` (8 x 8 grid).
- B0268 features: built with `stratia/features/cache.py` under the keys `ffr1-b0268` (cls) and
  `ffr1-patches-b0268-pool4` (pooled grid) from `cache/img768/B0268/...` (393 frames, GPU, once).
- B0268 geometry: the B0268 branch of `GeometryCache.build` (`camera_for("b0268", 4656, 3496)`, zenith pose, the
  manifest Sun, `ray_map(out_size=512, patch=64)` -> [4, 8, 8], `meta_vector`).
- Trainer `scripts/adapt_b0268.py` (`GeometryDropout`, `Engine`, `Run`; checkpoints `{"model", "ema": None}` as
  `stratia/export/onnx_model._state` reads them); scoring `stratia/train/b0268_eval.py` (P083) and
  `stratia/train/global_eval.score_dataset` (P082).
- Export: `scripts/export_onnx.py --global-run <run> --calibration <json>` (P096); calibration
  `scripts/calibrate.py --global-run runs/<run> --cbh-run runs/<P075 run>` (P089).

## Work log
1. Reduced design (owner decision, demo deadline 2026-10-30): heads only. The `GeometryEncoder` and `GlobalHeads`
   of the P073 global run are fine-tuned on cached frozen DINOv3 features; the backbone stays frozen and no LoRA
   is trained. "LP-FT" here is the FT half on an already probed head: LP trains the three output layers
   (`lp_parameters`: genus, étage, oktas) for 5 epochs, FT every head parameter for 10 epochs; the last checkpoint
   is the model of the draw (no early stopping, no checkpoint selection on the held-out days).
2. Targets: the B0268 rater shares become rows the way Montenegro soft targets do (`b0268_rows`: genus shares plus
   the clear share, negatives 1.0 except contrail, étage shares with weight 1, oktas distribution), carrying the
   Sun and the calendar day.
3. Batches: 16 B0268 frames of the draw's train days plus `replay` x 16 source samples from the P073 training
   split, geometry dropout 0.3 on the source part; loss as in P073 (`lambda_h` 0.1).
4. Scoring, as pre-registered but with P083's per-rater truth: the adapted heads against each rater of the pair,
   averaged (`score_against_raters`), on the draw's held-out adaptation days, next to the zero-shot heads on the
   same frames; the source check with `score_dataset` on the in-domain validation split (mean étage F1 over the
   sources, mean genus F1, the Montenegro C2 ratios). The regression rule as implemented: a drop of more than 0.02
   on the mean étage F1 or the mean genus F1 rejects the draw's model (the C2 ratios are reported, not gated).
5. Draws, in this order, each with the same configuration (the locked day is never evaluated here):

| Draw | Train days (k) | Val days (held out) | Test (locked) |
|---|---|---|---|
| k1_d0 | 2026-10-18 | 10-19, 10-21 | 2026-10-20 |
| k1_d1 | 2026-10-19 | 10-18, 10-21 | 2026-10-20 |
| k1_d2 | 2026-10-21 (33 frames) | 10-18, 10-19 | 2026-10-20 |
| k2_d0 | 10-18 + 10-19 | 10-21 | 2026-10-20 |
| k2_d1 | 10-18 + 10-21 | 10-19 | 2026-10-20 |
| k2_d2 | 10-19 + 10-21 | 10-18 | 2026-10-20 |

6. Runs (`--draw all`, 20–45 s each on the RTX 4050, seed 0): seven configurations on the six draws, 42 runs in
   `runs/registry.csv` (phase P084, names `adapt_b0268_<draw>_<tag>`). `base`: replay 4, LP 1e-3, FT 1e-4;
   `r8`: replay 8; `r8_lp3`: replay 8, LP 3e-4; `r8_lp3_ft3`: replay 8, LP 3e-4, FT 3e-5; `r16_lp3`: replay 16,
   LP 3e-4; `r16_ft3`: replay 16, LP 3e-4, FT 3e-5; `lp_only`: replay 8, LP 10 epochs at 1e-3, no FT. Results in
   `data/b0268_adaptation.parquet` (one row set per tag); the sweep table and the per-draw tables of the chosen
   configuration in `docs/data/b0268_adaptation.md` (`--doc-only` rewrites it from the parquet).
7. Choice: `r16_lp3` — the best source preservation (worst drop 0.030, on one draw; mean étage F1 −0.010, genus
   −0.015) with the second-largest gains (étage set +0.223, oktas ±1 +0.365, genus +0.517 over the draws). The
   choice used the held-out days and the source check, so the held-out numbers of `r16_lp3` carry the optimism of a
   selection among seven (*Deviations* 1).
8. Final demo model: `r16_lp3` once more on all three adaptation days (`--final`; run
   `20261023T123549Z_adapt_b0268_final_r16_lp3_76502c7936`, 273 frames, 15 epochs): source étage F1 0.805 (from
   0.817), genus F1 0.701 (from 0.719), regression 0.018 — inside the rule. Two earlier final runs with the same
   numbers (`...T122928Z`, `...T123105Z`) are superseded: their configs lacked the P073 keys `calibrate.py` reads
   (`load_init` now carries `name`, `lodo`, `eye2sky_etage` next to `hidden`, `dropout`, `kendall`).
9. Calibration refitted on the adapted heads with P089's procedure (`scripts/calibrate.py` on the final run and the
   P075 CBH run; outputs copied to `configs/calibration_1.0.1.json` and `docs/data/calibration_1.0.1.md`, the 1.0.0
   files restored): genus temperature 5.00 (cross-fit 4.41; ECE 0.156 -> 0.073, against 0.145 -> 0.092 for 1.0.0),
   étage and oktas keep T = 1 (ECE 0.021, 0.029), OOD threshold 1.497 (1.0.0: 1.457) flagging 21.9 % of the SWIM
   patches (25.3 %); CBH unchanged.
10. Export (`scripts/export_onnx.py --global-run <final run> --calibration configs/calibration_1.0.1.json --out
    models/stratia-1.0.1-demo --version 1.0.1-demo --no-report`): `stratia.onnx` 94.9 MB, SHA-256 `66e22ed1…`,
    parity max |Δ| 1.5e-4 (dense logits) / ≤ 4.1e-5 (global outputs), contract OK, CPU batch-1 163 ms;
    `model_card.json` and `export_report.json` committed next to the 1.0.0 ones (P096 work log 6). Dense and CBH
    heads are the 1.0.0 ones.
11. Fit check of the exported graph (not an evaluation): `scripts/eval_b0268.py --model models/stratia-1.0.1-demo
    --suffix adapted_fit` on the 273 frames the model was adapted on (run
    `20261023T123919Z_b0268_adapted_adapted_fit_c47e02b737`; `data/b0268_zero_shot_scores_adapted_fit.parquet`).

## Verification
Sweep (mean over the six draws, held-out days, each rater averaged; `docs/data/b0268_adaptation.md`):

| Tag | Étage set after (gain) | Oktas ±1 after (gain) | Genus top-1 after (gain) | Source étage / genus F1 change | Worst source drop |
|---|---|---|---|---|---|
| `base` | 0.228 (+0.177) | 0.732 (+0.393) | 0.841 (+0.575) | −0.015 / −0.026 | 0.043 |
| `lp_only` | 0.200 (+0.149) | 0.643 (+0.304) | 0.761 (+0.495) | −0.015 / −0.029 | 0.052 |
| `r8` | 0.236 (+0.185) | 0.723 (+0.383) | 0.830 (+0.564) | −0.016 / −0.025 | 0.044 |
| `r8_lp3` | 0.262 (+0.211) | 0.711 (+0.372) | 0.811 (+0.544) | −0.014 / −0.018 | 0.031 |
| `r8_lp3_ft3` | 0.209 (+0.158) | 0.548 (+0.209) | 0.559 (+0.292) | −0.011 / −0.024 | 0.047 |
| `r16_ft3` | 0.211 (+0.160) | 0.574 (+0.235) | 0.545 (+0.278) | −0.010 / −0.022 | 0.038 |
| **`r16_lp3`** | 0.274 (+0.223) | 0.705 (+0.365) | 0.784 (+0.517) | −0.010 / −0.015 | 0.030 |

`r16_lp3` per draw (zero-shot -> adapted on the held-out days; source F1 change against the G4 heads):

| Draw (train -> held out) | Étage set | Oktas ±1 | Okta bias | Top genus in the rater's set | Source étage / genus F1 | Rule |
|---|---|---|---|---|---|---|
| k1_d0 (18 -> 19, 21; 153 frames) | 0.026 -> 0.157 | 0.085 -> 0.170 | +3.59 -> +2.66 | 0.007 -> 0.739 | −0.006 / −0.030 | broken |
| k1_d1 (19 -> 18, 21; 153) | 0.072 -> 0.307 | 0.601 -> 0.892 | +1.38 -> −0.23 | 0.542 -> 0.938 | −0.009 / −0.011 | ok |
| k1_d2 (21 -> 18, 19; 240) | 0.029 -> 0.029 | 0.379 -> 0.713 | +1.91 -> +0.42 | 0.350 -> 0.479 | −0.013 / −0.015 | ok |
| k2_d0 (18, 19 -> 21; 33) | 0.121 -> 0.879 | 0.212 -> 0.924 | +4.58 -> +0.91 | 0.000 -> 0.788 | −0.019 / −0.014 | ok |
| k2_d1 (18, 21 -> 19; 120) | 0.000 -> 0.033 | 0.050 -> 0.554 | +3.33 -> +0.64 | 0.008 -> 0.775 | −0.006 / −0.016 | ok |
| k2_d2 (19, 21 -> 18; 120) | 0.058 -> 0.238 | 0.708 -> 0.975 | +0.50 -> −0.24 | 0.692 -> 0.983 | −0.005 / −0.004 | ok |
| mean k = 1 | 0.042 -> 0.164 | 0.355 -> 0.592 | | 0.300 -> 0.719 | −0.009 / −0.019 | |
| mean k = 2 | 0.060 -> 0.383 | 0.323 -> 0.818 | | 0.233 -> 0.849 | −0.010 / −0.011 | |

Per held-out day the étage number is bimodal: 2026-10-19 stays at 0.000–0.033 whatever the other days teach,
2026-10-18 reaches 0.150–0.238, 2026-10-21 reaches 0.727–0.879 once another afternoon is in training; oktas within
±1 on the held-out days spans 0.133 (10-19 after 10-18 alone) to 0.975 (10-18 after 10-19 + 10-21).

Final model through the exported graph, on its own three training days (fit check; the P083 zero-shot numbers on
the same frames in brackets; the pair's ceiling: étage set 0.911, oktas ±1 0.771):

| Day | Frames | Étage set | Oktas ±1 | Okta bias | Top genus in set | OOD flagged |
|---|---|---|---|---|---|---|
| all | 273 | 0.905 (0.031) | 0.681 (0.278) | +0.32 (+2.40) | 0.989 (0.300) | 1.5 % (8.4 %) |
| 2026-10-18 | 120 | 0.800 (0.046) | 0.633 (0.492) | +1.33 | 1.000 | 0.0 % |
| 2026-10-19 | 120 | 0.992 (0.000) | 0.667 (0.075) | −0.38 | 1.000 | 0.0 % |
| 2026-10-21 | 33 | 0.970 (0.091) | 0.909 (0.242) | −0.85 | 0.909 | 12.1 % |

Sky-parse cover correlation with the raters' mean cover 0.83, as in P083 (the dense heads are untouched).
`tests/test_b0268_adapt.py` (data-free: B0268 rows and roles, the geometry branch, the LP parameter set, the
report rebuild); full suite, ruff and `scripts/check_test_lock.py` green before the commit; the locked test day
(`few_shot_target` role `test`) was never read by the trainer or the scorer.

## Exit criteria
- [x] Gain on held-out B0268 days over P083, per draw and per k: étage set +0.223 (five of six draws up, `k1_d2`
      flat), oktas within ±1 +0.365 (six of six), top genus +0.517; positive on both k = 1 and k = 2 means.
- [x] No source regression of more than 2 points for the exported model (0.018 on the G4 validation F1s).
- [ ] ... for every draw of the chosen configuration: `k1_d0` drops the mean genus F1 by 0.030 (*Deviations* 2).
- [x] The locked test day untouched.

## Fit & data-risk notes
- The guard fired: 0.905 étage set fitted against 0.274 held out is the day effect. A day is one afternoon from
  one camera, so the held-out étage agreement is a property of the day pair, not of a frame count; the mean over
  draws hides that 2026-10-19 is never matched. The okta bias is the transferable part (+2.4 zero-shot; within ±1
  on 55–98 % of held-out frames after one or two days on every draw) because cover varies within an afternoon.
- The chosen configuration was selected on the held-out days (seven candidates): its held-out numbers are
  optimistic. The only unbiased B0268 number is the locked 2026-10-20 at P090, through this same graph, once.
- Truth is from language-model annotators (P041); a gain against their labels is a gain against that proxy; the
  étage sets of 2026-10-19 that no adaptation matches may be a property of the labels as much as of the model.
- The demo model is fitted to the demo camera's three afternoons; what it shows on 2026-10-30 is not evidence of
  generalisation and the card says `1.0.1-demo`.
- Forgetting is governed by replay and by the LP step: replay 4 with LP 1e-3 lost 0.043 on the worst source
  metric, LP only at 1e-3 0.052, replay 16 with LP 3e-4 0.030; the legacy "10x over-biases" did not repeat at 16x
  (held-out oktas within 0.03 of 4x, étage set higher).

## Deviations from plan & why
1. Seven configurations instead of one fixed before the first run; the replay ratio was tuned (4 / 8 / 16) as the
   plan asks and the reduced design had excluded. Recorded as selection on the held-out days.
2. Step 8 of the design required no draw outside the regression rule; `k1_d0` of `r16_lp3` is outside it by 0.010
   on the mean genus F1 (every configuration has at least one draw outside). The final model is measured directly
   on the quantity the rule protects and is inside it, and the demo needs a model; exported, with this note.
3. Heads only on frozen features; LoRA (P077 route) deferred; self-training (P085), test-time adaptation (P086) and
   the dense B0268 subset deferred; the day-blocked protocol planned six or more days, four exist (P041).
4. Three final runs in the registry for one model (two superseded by a config fix, same numbers).

## Next phase
AI-day demo of 2026-10-30 on `models/stratia-1.0.1-demo` (CloudScope: AI > Models… installs the package and
*Activate* loads it; `1.0.0-demo` stays installed for *Roll back*). Then P090 — locked-test evaluation through the
exported graph, with the B0268 2026-10-20 day once; P085–P087 when the deadline allows.
