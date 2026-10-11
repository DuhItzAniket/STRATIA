# OOD score evaluation (P068)

`OodScore` (energy on a genus probe's logits + Mahalanobis on the pooled dinov3_vits16 @ 512 feature, z-scored on the training rows, averaged) in a leave-one-source-out design: fitted on two sources' train rows, AUROC of the held-out source's val rows (out) against the training sources' val rows (in). The last two columns score the same in-domain rows against 2,000 Eye2Sky all-sky frames and 2,000 SWIM patches.

| Held-out source | n in / out | AUROC combined | energy only | Mahalanobis only | vs Eye2Sky | vs SWIM patches |
|---|---|---|---|---|---|---|
| ccsn | 1470 / 255 | 0.999 | 0.455 | 1.000 | 0.996 | 0.998 |
| mgcd | 766 / 959 | 0.961 | 0.872 | 0.963 | 0.991 | 0.874 |
| montenegro | 1214 / 511 | 0.974 | 0.755 | 0.968 | 0.939 | 0.911 |
