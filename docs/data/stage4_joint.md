# Stage-4 joint multi-task training on cached features (P076)

All heads of the STRATIA model trained together from cached DINOv3 ViT-S/16 features with the P069 sampler (global / dense / cbh batches), the P070 weighting and the gradient-cosine monitor; the single-task reference is the best run of P073 (indomain + Eye2Sky étage), P074 (dense + layers, Kendall) and P075 (CBH base). Validation numbers only.

| Head / dataset / metric | Joint | Single-task | Δ |
|---|---|---|---|
| **joint** (best epoch 12) | | | |
| cbh/all/cloud_balanced_acc | 0.924 | 0.930 | -0.005 |
| cbh/all/coverage_80 | 0.936 | 0.927 | +0.009 |
| cbh/all/crps | 0.029 | 0.026 | +0.003 |
| cbh/all/mae_m | 618.193 | 546.239 | +71.955 |
| cbh/all/skill | 0.461 | 0.524 | -0.063 |
| dense/all/miou | 0.870 | 0.905 | -0.035 |
| dense/almeria/layer_miou | 0.544 | 0.559 | -0.015 |
| dense/almeria/miou | 0.866 | 0.885 | -0.020 |
| dense/shwimseg/miou | 0.778 | 0.809 | -0.030 |
| dense/swimseg/miou | 0.830 | 0.879 | -0.049 |
| dense/swinseg/miou | 0.883 | 0.903 | -0.020 |
| dense/swinyseg/miou | 0.837 | 0.878 | -0.040 |
| global/almeria/etage_f1 | 0.788 | 0.758 | +0.029 |
| global/almeria/etage_unsupported_fp | nan | nan | +nan |
| global/almeria/n_etage | 84.000 | 84.000 | +0.000 |
| global/ccsn/etage_f1 | 0.796 | 0.771 | +0.026 |
| global/ccsn/etage_unsupported_fp | nan | nan | +nan |
| global/ccsn/genus_acc | 0.667 | 0.647 | +0.020 |
| global/ccsn/genus_f1 | 0.652 | 0.642 | +0.011 |
| global/ccsn/n_etage | 236.000 | 236.000 | +0.000 |
| global/ccsn/n_genus | 255.000 | 255.000 | +0.000 |
| global/eye2sky_cbh/etage_recall | 1.000 | 1.000 | +0.000 |
| global/eye2sky_cbh/n_etage | 1233.000 | 1233.000 | +0.000 |
| global/mgcd/etage_f1 | 1.000 | 1.000 | +0.000 |
| global/mgcd/etage_unsupported_fp | 0.000 | 0.000 | +0.000 |
| global/mgcd/genus_acc | 0.820 | 0.840 | -0.020 |
| global/mgcd/genus_f1 | 0.779 | 0.799 | -0.019 |
| global/mgcd/n_etage | 438.000 | 438.000 | +0.000 |
| global/mgcd/n_genus | 789.000 | 789.000 | +0.000 |
| global/montenegro/c2_etage_ratio | 1.034 | 1.045 | -0.011 |
| global/montenegro/c2_oktas_ratio | 0.973 | 1.045 | -0.072 |
| global/montenegro/etage_f1 | 0.666 | 0.739 | -0.073 |
| global/montenegro/etage_set_agreement | 0.881 | 0.890 | -0.010 |
| global/montenegro/etage_unsupported_fp | nan | nan | +nan |
| global/montenegro/n_etage | 511.000 | 511.000 | +0.000 |
| global/montenegro/oktas_kl | 0.646 | 0.565 | +0.081 |
| global/montenegro/oktas_mae | 0.595 | 0.458 | +0.137 |
| global/montenegro/oktas_within1 | 0.849 | 0.912 | -0.063 |
| global/swimcat/etage_recall | 0.000 | 0.000 | +0.000 |
| global/swimcat/n_etage | 28.000 | 28.000 | +0.000 |
| **joint_long** (best epoch 7) | | | |
| cbh/all/cloud_balanced_acc | 0.927 | 0.930 | -0.003 |
| cbh/all/coverage_80 | 0.930 | 0.927 | +0.003 |
| cbh/all/crps | 0.026 | 0.026 | -0.000 |
| cbh/all/mae_m | 545.103 | 546.239 | -1.136 |
| cbh/all/skill | 0.525 | 0.524 | +0.001 |
| dense/all/miou | 0.881 | 0.905 | -0.024 |
| dense/almeria/layer_miou | 0.421 | 0.559 | -0.138 |
| dense/almeria/miou | 0.862 | 0.885 | -0.024 |
| dense/shwimseg/miou | 0.801 | 0.809 | -0.007 |
| dense/swimseg/miou | 0.848 | 0.879 | -0.031 |
| dense/swinseg/miou | 0.892 | 0.903 | -0.011 |
| dense/swinyseg/miou | 0.852 | 0.878 | -0.025 |
| global/almeria/etage_f1 | 0.746 | 0.758 | -0.012 |
| global/almeria/etage_unsupported_fp | nan | nan | +nan |
| global/almeria/n_etage | 84.000 | 84.000 | +0.000 |
| global/ccsn/etage_f1 | 0.760 | 0.771 | -0.011 |
| global/ccsn/etage_unsupported_fp | nan | nan | +nan |
| global/ccsn/genus_acc | 0.686 | 0.647 | +0.039 |
| global/ccsn/genus_f1 | 0.679 | 0.642 | +0.037 |
| global/ccsn/n_etage | 236.000 | 236.000 | +0.000 |
| global/ccsn/n_genus | 255.000 | 255.000 | +0.000 |
| global/eye2sky_cbh/etage_recall | 0.972 | 1.000 | -0.028 |
| global/eye2sky_cbh/lowest_etage_acc | 0.843 | - | - |
| global/eye2sky_cbh/n_etage | 1233.000 | 1233.000 | +0.000 |
| global/mgcd/etage_f1 | 1.000 | 1.000 | +0.000 |
| global/mgcd/etage_unsupported_fp | 0.000 | 0.000 | +0.000 |
| global/mgcd/genus_acc | 0.866 | 0.840 | +0.025 |
| global/mgcd/genus_f1 | 0.819 | 0.799 | +0.020 |
| global/mgcd/n_etage | 438.000 | 438.000 | +0.000 |
| global/mgcd/n_genus | 789.000 | 789.000 | +0.000 |
| global/montenegro/c2_etage_ratio | 1.029 | 1.045 | -0.016 |
| global/montenegro/c2_oktas_ratio | 1.042 | 1.045 | -0.002 |
| global/montenegro/etage_f1 | 0.691 | 0.739 | -0.048 |
| global/montenegro/etage_set_agreement | 0.877 | 0.890 | -0.014 |
| global/montenegro/etage_unsupported_fp | nan | nan | +nan |
| global/montenegro/n_etage | 511.000 | 511.000 | +0.000 |
| global/montenegro/oktas_kl | 0.574 | 0.565 | +0.009 |
| global/montenegro/oktas_mae | 0.466 | 0.458 | +0.008 |
| global/montenegro/oktas_within1 | 0.910 | 0.912 | -0.002 |

## Task-gradient cosines on the shared parameters (mean over logged steps)

| Pair | Mean cosine | Min |
|---|---|---|
| cbh|etage | +0.012 | -0.170 |
| cbh|genus | -0.000 | -0.035 |
| cbh|layer | -0.013 | -0.287 |
| cbh|oktas | -0.000 | -0.024 |
| cbh|sky | -0.000 | -0.147 |
| etage|genus | +0.040 | -0.021 |
| etage|layer | +0.001 | -0.085 |
| etage|oktas | +0.006 | -0.049 |
| etage|sky | -0.001 | -0.086 |
| genus|layer | +0.001 | -0.076 |
| genus|oktas | +0.027 | -0.073 |
| genus|sky | -0.000 | -0.055 |
| layer|oktas | +0.002 | -0.033 |
| layer|sky | +0.005 | -0.921 |
| oktas|sky | +0.000 | -0.046 |

Rows in `data/stage4_joint.parquet`; curves and cosines in the run folder.
