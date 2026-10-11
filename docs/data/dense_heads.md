# Dense heads on cached DINOv3 tokens (P065 / P066)

DPT-lite decoder (ADR-005 §2) on the frozen DINOv3 ViT-S/16 32 x 32 token grid at 512 px, grouped train -> val (test locked), masks at 512 px, mIoU over the classes present in each dataset's truth (as P057), 95 % bootstrap intervals over val images, and the clear-sky collapse check (predicted-cloud fraction inside the camera mask on 200 clear-sky Eye2Sky frames). The P057 token probe is the reference the decoder has to beat.

| Phase | Model | Dataset | mIoU [95 % CI] | IoU sky | IoU cloud | IoU invalid | Layer mIoU [CI] | low / mid / high IoU | Clear-sky cloud fraction | Probe mIoU |
|---|---|---|---|---|---|---|---|---|---|---|
| P065 | dense_heads | all | 0.919 [0.916, 0.923] | 0.887 | 0.897 | 0.974 | - | - / - / - | 0.220 | 0.875 |
| P065 | dense_heads | almeria | 0.902 [0.884, 0.918] | 0.883 | 0.847 | 0.974 | - | - / - / - | 0.220 | 0.865 |
| P065 | dense_heads | shwimseg | 0.833 [0.755, 0.890] | 0.807 | 0.858 | - | - | - / - / - | 0.220 | 0.786 |
| P065 | dense_heads | swimseg | 0.892 [0.875, 0.909] | 0.888 | 0.897 | - | - | - / - / - | 0.220 | 0.847 |
| P065 | dense_heads | swinseg | 0.911 [0.891, 0.928] | 0.917 | 0.904 | - | - | - / - / - | 0.220 | 0.866 |
| P065 | dense_heads | swinyseg | 0.895 [0.889, 0.901] | 0.889 | 0.902 | - | - | - / - / - | 0.220 | 0.843 |
| P066 | dense_heads+layers | all | 0.894 [0.889, 0.899] | 0.860 | 0.870 | 0.953 | 0.559 [0.463, 0.659] | 0.575 / 0.344 / 0.758 | 0.357 | 0.875 |
| P066 | dense_heads+layers | almeria | 0.873 [0.854, 0.892] | 0.856 | 0.810 | 0.953 | 0.559 [0.467, 0.661] | 0.575 / 0.344 / 0.758 | 0.357 | 0.865 |
| P066 | dense_heads+layers | shwimseg | 0.799 [0.747, 0.845] | 0.768 | 0.830 | - | - | - / - / - | 0.357 | 0.786 |
| P066 | dense_heads+layers | swimseg | 0.868 [0.848, 0.888] | 0.864 | 0.871 | - | - | - / - / - | 0.357 | 0.847 |
| P066 | dense_heads+layers | swinseg | 0.898 [0.882, 0.912] | 0.905 | 0.890 | - | - | - / - / - | 0.357 | 0.866 |
| P066 | dense_heads+layers | swinyseg | 0.868 [0.860, 0.876] | 0.861 | 0.875 | - | - | - / - / - | 0.357 | 0.843 |

Rows in `data/dense_heads.parquet`; curves in each run's `metrics.jsonl` (phase P065 / P066).
