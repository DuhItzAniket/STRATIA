# P078 — Budgeted HPO

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
A small, logged hyper-parameter search for the global heads on cached features, validation only, with a
sensitivity read: which knobs matter and whether the defaults are near the top of the sampled space.

## Inputs / dependencies
P073 trainer and in-domain split (with Eye2Sky étage rows); P040 locked test (not read); PLAN guard "val overfitting
through HPO — keep the budget small, report it".

## Work log
1. `scripts/hpo_global.py`: seeded random search over learning rate (log 2e-4–3e-3), weight decay (log 1e-3–0.3),
   trunk dropout (0–0.4), hidden width {256, 512, 1024}, sampler temperature (0.3–1.0), geometry dropout
   {0, 0.15, 0.3} and the hierarchy weight {0, 0.1, 0.3}; every trial is one `train_global_heads.py` run of 12 epochs
   (patience 4) named `hpo_t<i>`, scored by its best val mean étage macro-F1; `data/hpo_global.parquet`,
   `docs/data/hpo_global.md` with the trials, the best configuration and the Spearman correlation of each
   parameter with the score. Optuna is not in the environment lock; a plain random search is what a 20-trial
   budget needs.
2. 20 trials, 3–4 minutes each (70 minutes).

## Verification
- Scores 0.775–0.819 (median 0.798); the P073 default configuration (lr 1e-3, weight decay 0.05, dropout 0.1,
  hidden 512, τ 0.5, geometry dropout 0.3, λ_h 0.1) scores 0.817 under the same 20-epoch protocol and 0.80–0.81
  under the 12-epoch one.
- Best trial: lr 1.28e-3, weight decay 0.041, dropout 0.28, hidden 1024, τ 0.57, geometry dropout 0.3, λ_h 0,
  score 0.819.
- Sensitivity (Spearman with the score over 20 trials): learning rate +0.35, hidden +0.20, weight decay −0.20,
  dropout −0.09, λ_h −0.08, τ −0.01, geometry dropout +0.02. Nothing is strongly sensitive; the top and the
  default are 0.002 apart, which is below the seed std measured in P080.

## Exit criteria
- [x] Best config + sensitivity table (`docs/data/hpo_global.md`); budget reported: 20 trials, validation only.

## Fit & data-risk notes
- Twenty configurations evaluated on the same validation split is twenty looks at val; the gain found is 0.002,
  so no configuration change is adopted from this search and nothing was tuned toward val beyond the P073 defaults.
  The locked test evaluation (P090) will report this budget.
- λ_h = 0 appearing in the best trial says the hierarchy term is not helping in-domain; its purpose is the LODO
  étage consistency, where P080's LODO variants test it directly.

## Deviations from plan & why
- Random search instead of Optuna (not in the lock); 20 of the plan's ≤ 30 trials, because the first 20 already
  show a flat response.

## Next phase
P079 — Learning curves.
