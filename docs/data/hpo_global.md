# Budgeted hyper-parameter search, global heads (P078)

Random search, 20 trials of 12 epochs each on cached features (in-domain split with Eye2Sky étage rows), scored by the best val mean étage macro-F1; the locked test sets are not read. Sensitivity = Spearman rank correlation of each parameter with the score over the trials (a one-shot read of the sampled space, not a response surface).

| Trial | lr | weight decay | dropout | hidden | tau | p_drop | lambda_h | Score | s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.12e-03 | 0.005 | 0.02 | 256 | 0.87 | 0.0 | 0.1 | 0.801 | 270 |
| 1 | 1.03e-03 | 0.064 | 0.22 | 1024 | 0.95 | 0.0 | 0.3 | 0.809 | 260 |
| 2 | 2.01e-04 | 0.133 | 0.01 | 1024 | 0.42 | 0.3 | 0.0 | 0.793 | 255 |
| 3 | 8.67e-04 | 0.006 | 0.17 | 1024 | 0.32 | 0.0 | 0.0 | 0.814 | 255 |
| 4 | 1.23e-03 | 0.040 | 0.25 | 1024 | 1.00 | 0.15 | 0.3 | 0.794 | 178 |
| 5 | 1.28e-03 | 0.041 | 0.28 | 1024 | 0.57 | 0.3 | 0.0 | 0.819 | 243 |
| 6 | 1.41e-03 | 0.020 | 0.12 | 512 | 0.92 | 0.15 | 0.0 | 0.797 | 206 |
| 7 | 5.27e-04 | 0.026 | 0.13 | 1024 | 0.72 | 0.15 | 0.1 | 0.798 | 172 |
| 8 | 5.78e-04 | 0.160 | 0.09 | 1024 | 0.36 | 0.15 | 0.1 | 0.812 | 208 |
| 9 | 1.69e-03 | 0.004 | 0.35 | 1024 | 0.34 | 0.3 | 0.1 | 0.810 | 211 |
| 10 | 3.00e-04 | 0.013 | 0.32 | 1024 | 0.34 | 0.0 | 0.1 | 0.793 | 172 |
| 11 | 3.42e-04 | 0.002 | 0.23 | 512 | 0.51 | 0.3 | 0.3 | 0.795 | 205 |
| 12 | 3.43e-04 | 0.216 | 0.15 | 512 | 0.74 | 0.0 | 0.3 | 0.775 | 206 |
| 13 | 6.59e-04 | 0.232 | 0.20 | 1024 | 0.60 | 0.3 | 0.1 | 0.811 | 209 |
| 14 | 2.96e-03 | 0.224 | 0.18 | 1024 | 0.65 | 0.3 | 0.1 | 0.788 | 209 |
| 15 | 1.68e-03 | 0.011 | 0.29 | 512 | 0.80 | 0.3 | 0.3 | 0.798 | 209 |
| 16 | 2.73e-04 | 0.064 | 0.37 | 1024 | 0.31 | 0.3 | 0.0 | 0.787 | 186 |
| 17 | 2.85e-03 | 0.235 | 0.06 | 1024 | 0.98 | 0.15 | 0.3 | 0.798 | 211 |
| 18 | 1.85e-03 | 0.015 | 0.09 | 1024 | 0.95 | 0.3 | 0.3 | 0.814 | 227 |
| 19 | 8.61e-04 | 0.012 | 0.37 | 256 | 0.33 | 0.3 | 0.3 | 0.799 | 220 |

**Best:** trial 5 — score 0.819: lr 1.28e-03, weight decay 0.041, dropout 0.28, hidden 1024, tau 0.57, p_drop 0.3, lambda_h 0.0. Spread over trials: 0.775–0.819 (median 0.798).

| Parameter | Spearman with score |
|---|---|
| lr | +0.35 |
| weight_decay | -0.20 |
| dropout | -0.09 |
| hidden | +0.20 |
| tau | -0.01 |
| p_drop | +0.02 |
| lambda_h | -0.08 |

Rows in `data/hpo_global.parquet`; each trial is a run of `scripts/train_global_heads.py` (phase P078).
