# Stage-5 partial unfreeze: LoRA on the backbone (P077)

DINOv3 ViT-S/16 with LoRA adapters on the q / k / v projections of the last blocks, heads from the P076 joint checkpoint, P048 augmentation, bfloat16, gradient checkpointing. 'Before' = the same heads on the frozen backbone evaluated through this image pipeline (adapters disabled); 'after' = the best epoch. A negative Δ on a dataset the run did not improve is forgetting.

| Run | Metric | Before (frozen) | After (LoRA) | Δ |
|---|---|---|---|---|
| lora (epoch 2) | dense/all/miou | 0.869 | 0.878 | +0.009 |
| lora (epoch 2) | dense/almeria/layer_miou | 0.517 | 0.471 | -0.045 |
| lora (epoch 2) | dense/almeria/miou | 0.862 | 0.864 | +0.001 |
| lora (epoch 2) | dense/shwimseg/miou | 0.798 | 0.787 | -0.011 |
| lora (epoch 2) | dense/swimseg/miou | 0.831 | 0.847 | +0.016 |
| lora (epoch 2) | dense/swinseg/miou | 0.883 | 0.884 | +0.000 |
| lora (epoch 2) | dense/swinyseg/miou | 0.839 | 0.849 | +0.010 |
| lora (epoch 2) | global/almeria/etage_f1 | 0.799 | 0.784 | -0.015 |
| lora (epoch 2) | global/almeria/n_etage | 84.000 | 84.000 | +0.000 |
| lora (epoch 2) | global/ccsn/etage_f1 | 0.784 | 0.736 | -0.049 |
| lora (epoch 2) | global/ccsn/genus_acc | 0.663 | 0.620 | -0.043 |
| lora (epoch 2) | global/ccsn/genus_f1 | 0.644 | 0.599 | -0.046 |
| lora (epoch 2) | global/ccsn/n_etage | 236.000 | 236.000 | +0.000 |
| lora (epoch 2) | global/ccsn/n_genus | 255.000 | 255.000 | +0.000 |
| lora (epoch 2) | global/mean_etage_f1 | 0.805 | 0.810 | +0.005 |
| lora (epoch 2) | global/mgcd/etage_f1 | 1.000 | 1.000 | +0.000 |
| lora (epoch 2) | global/mgcd/etage_unsupported_fp | 0.000 | 0.000 | +0.000 |
| lora (epoch 2) | global/mgcd/genus_acc | 0.825 | 0.790 | -0.035 |
| lora (epoch 2) | global/mgcd/genus_f1 | 0.786 | 0.754 | -0.031 |
| lora (epoch 2) | global/mgcd/n_etage | 438.000 | 438.000 | +0.000 |
| lora (epoch 2) | global/mgcd/n_genus | 789.000 | 789.000 | +0.000 |
| lora (epoch 2) | global/montenegro/c2_etage_ratio | 1.018 | 1.047 | +0.030 |
| lora (epoch 2) | global/montenegro/c2_oktas_ratio | 0.980 | 0.991 | +0.011 |
| lora (epoch 2) | global/montenegro/etage_f1 | 0.638 | 0.719 | +0.082 |
| lora (epoch 2) | global/montenegro/etage_set_agreement | 0.867 | 0.892 | +0.025 |
| lora (epoch 2) | global/montenegro/n_etage | 511.000 | 511.000 | +0.000 |
| lora (epoch 2) | global/montenegro/oktas_kl | 0.646 | 0.621 | -0.025 |
| lora (epoch 2) | global/montenegro/oktas_mae | 0.589 | 0.571 | -0.018 |
| lora (epoch 2) | global/montenegro/oktas_within1 | 0.855 | 0.865 | +0.010 |
| lora_lowhead (epoch 1) | dense/all/miou | 0.869 | 0.877 | +0.008 |
| lora_lowhead (epoch 1) | dense/almeria/layer_miou | 0.517 | 0.509 | -0.008 |
| lora_lowhead (epoch 1) | dense/almeria/miou | 0.862 | 0.868 | +0.006 |
| lora_lowhead (epoch 1) | dense/shwimseg/miou | 0.798 | 0.806 | +0.008 |
| lora_lowhead (epoch 1) | dense/swimseg/miou | 0.831 | 0.837 | +0.006 |
| lora_lowhead (epoch 1) | dense/swinseg/miou | 0.883 | 0.884 | +0.000 |
| lora_lowhead (epoch 1) | dense/swinyseg/miou | 0.839 | 0.846 | +0.007 |
| lora_lowhead (epoch 1) | global/almeria/etage_f1 | 0.799 | 0.795 | -0.003 |
| lora_lowhead (epoch 1) | global/almeria/etage_f1_tuned | 0.829 | 0.823 | -0.006 |
| lora_lowhead (epoch 1) | global/almeria/n_etage | 84.000 | 84.000 | +0.000 |
| lora_lowhead (epoch 1) | global/ccsn/etage_f1 | 0.784 | 0.773 | -0.011 |
| lora_lowhead (epoch 1) | global/ccsn/etage_f1_tuned | 0.806 | 0.791 | -0.015 |
| lora_lowhead (epoch 1) | global/ccsn/genus_acc | 0.663 | 0.635 | -0.027 |
| lora_lowhead (epoch 1) | global/ccsn/genus_f1 | 0.644 | 0.624 | -0.020 |
| lora_lowhead (epoch 1) | global/ccsn/n_etage | 236.000 | 236.000 | +0.000 |
| lora_lowhead (epoch 1) | global/ccsn/n_genus | 255.000 | 255.000 | +0.000 |
| lora_lowhead (epoch 1) | global/mean_etage_f1 | 0.805 | 0.810 | +0.005 |
| lora_lowhead (epoch 1) | global/mgcd/etage_f1 | 1.000 | 1.000 | +0.000 |
| lora_lowhead (epoch 1) | global/mgcd/etage_f1_tuned | 1.000 | 1.000 | +0.000 |
| lora_lowhead (epoch 1) | global/mgcd/etage_unsupported_fp | 0.000 | 0.000 | +0.000 |
| lora_lowhead (epoch 1) | global/mgcd/genus_acc | 0.825 | 0.852 | +0.027 |
| lora_lowhead (epoch 1) | global/mgcd/genus_f1 | 0.786 | 0.812 | +0.026 |
| lora_lowhead (epoch 1) | global/mgcd/n_etage | 438.000 | 438.000 | +0.000 |
| lora_lowhead (epoch 1) | global/mgcd/n_genus | 789.000 | 789.000 | +0.000 |
| lora_lowhead (epoch 1) | global/montenegro/c2_etage_ratio | 1.018 | 1.038 | +0.021 |
| lora_lowhead (epoch 1) | global/montenegro/c2_oktas_ratio | 0.980 | 1.004 | +0.025 |
| lora_lowhead (epoch 1) | global/montenegro/etage_f1 | 0.638 | 0.671 | +0.033 |
| lora_lowhead (epoch 1) | global/montenegro/etage_f1_tuned | 0.679 | 0.682 | +0.003 |
| lora_lowhead (epoch 1) | global/montenegro/etage_set_agreement | 0.867 | 0.885 | +0.018 |
| lora_lowhead (epoch 1) | global/montenegro/n_etage | 511.000 | 511.000 | +0.000 |
| lora_lowhead (epoch 1) | global/montenegro/oktas_kl | 0.646 | 0.603 | -0.043 |
| lora_lowhead (epoch 1) | global/montenegro/oktas_mae | 0.589 | 0.524 | -0.065 |
| lora_lowhead (epoch 1) | global/montenegro/oktas_within1 | 0.855 | 0.877 | +0.022 |

Rows in `data/stage5_lora.parquet`; per-epoch curves in the run folder.
