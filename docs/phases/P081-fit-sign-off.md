# P081 — Fit sign-off

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
One report for the Stage H candidates: train / val at the best epoch per head and dataset with the PLAN §4 verdict,
the curves, calibration (ECE per global head, the CBH interval coverage and what a temperature does to it), so that
the gate reads numbers whose fit is known.

## Inputs / dependencies
The candidate runs: `indomain_e2s` (P073, final rerun), the P074 dense run, the P075 base CBH run (checkpoint and
val predictions saved), `joint_long` (P076), `lora_lowhead` (P077); PLAN §4; research design §4 (ECE, PIT / coverage).

## Work log
1. `scripts/fit_report.py`: reads each run's `curves.json` / `metrics.jsonl` and `summary.json`, computes the
   train–val gap at the best epoch and the §4 verdict (overfit: gap > 15 points and val falling after the best
   epoch; overfit-leaning: gap only; underfit: train < 0.6; else ok); reloads the global checkpoint (EMA weights)
   and computes the expected calibration error (15 bins) of the genus and étage probabilities (every
   (sample, class) Bernoulli on exhaustively labelled rows) and of the oktas top-class confidence; for the CBH head
   it reads the saved val predictions and cross-fits a temperature over the validation days (fit on half the days
   to reach 80 % coverage, apply to the other half, both ways). `docs/data/fit_report.md`, `data/fit_report.parquet`.

## Verification
Global heads (`indomain_e2s`, best epoch 11):

| Head | Dataset | Train | Val | Gap | Verdict |
|---|---|---|---|---|---|
| étage | CCSN | 0.961 | 0.802 | +0.16 | overfit-leaning |
| étage | MGCD | 1.000 | 1.000 | 0.00 | ok |
| étage | Montenegro | 0.776 | 0.692 | +0.08 | ok |
| étage | Almería | 0.924 | 0.775 | +0.15 | ok |
| genus | CCSN | 0.876 | 0.626 | +0.25 | overfit-leaning |
| genus | MGCD | 0.998 | 0.812 | +0.19 | overfit (val fell after epoch 11) |

- Calibration on val: étage ECE 0.013 (1,352 rows), genus ECE 0.059 (895 fully labelled rows), oktas top-class ECE
  0.061: the étage head is calibrated in-domain, the genus head mildly over-confident (consistent with its
  train–val gap).
- Dense (P074): val mIoU rises to epoch 14 (0.905) and dips by 0.001 at 15; the Kendall-weighted train loss keeps
  falling (negative because of the log-variance terms); verdict ok, stopped at the right place.
- Joint (P076 long run): headline 0.688 → 0.735 at epoch 7, then flat / slightly down (0.727–0.731); the CBH part
  was still improving when early stopping fired on the mean.
- LoRA (P077 low-head run): headline 0.843 at epoch 1, then 0.838 / 0.837: a fine-tune that peaks immediately.
- CBH head (P075 base): 80 % interval coverage 0.917 raw; with the cross-fitted temperature (0.49, fitted on half
  the validation days and applied to the other half) 0.802, and the median's MAE 580 m after scaling (566 m raw).
  The distributions are too wide by a factor two in logit scale, and a single scalar fixes the coverage; P089 fits
  it on its own calibration split.

## Exit criteria
- [x] Report signed (`docs/data/fit_report.md`): the final candidates are the single-task global heads, the P074
  dense heads, the P075 CBH head (joint_long's CBH numbers are kept as the alternative), with the LoRA path parked
  for Stage I.

## Fit & data-risk notes
- The genus head overfits CCSN and MGCD (train 0.88–1.00 against val 0.63–0.81). P079 says more data is the
  remedy that works (the curve is still rising); P080 says the regularisers tried move val by ≤ 0.01. The paper's
  genus numbers come with this gap stated.
- The temperature cross-fit uses the validation days twice over (fit on half, apply to the other half); it is an
  estimate of what calibration will do, not the calibration itself.

## Deviations from plan & why
- None.

## Next phase
P082 — Gate G4.
