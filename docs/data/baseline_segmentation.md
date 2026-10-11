# Segmentation baselines (P057)

Sky parsing (invalid / sky / cloud) on the grouped val splits of the SWIMSEG family and Almería. U-Net: 1.9 M parameters from scratch at 256 px. Probe: linear classifier on frozen DINOv3 ViT-S/16 patch tokens at 512 px, logits upsampled. Collapse check: mean predicted-cloud fraction inside the camera mask on clear-sky Eye2Sky frames (ceilometer label none, confidence >= 0.9); the legacy prototype predicted 85-99 % cloud on such frames. SAM 3 deferred.

| Model | Seed | Dataset | mIoU | IoU invalid | IoU sky | IoU cloud | Clear-sky cloud fraction |
|---|---|---|---|---|---|---|---|
| dinov3_vits16_patch_probe | 0 | all | 0.875 | 0.941 | 0.836 | 0.849 | 0.323 |
| dinov3_vits16_patch_probe | 0 | almeria | 0.865 | 0.941 | 0.850 | 0.805 | 0.323 |
| dinov3_vits16_patch_probe | 0 | shwimseg | 0.786 | - | 0.747 | 0.825 | 0.323 |
| dinov3_vits16_patch_probe | 0 | swimseg | 0.847 | - | 0.842 | 0.852 | 0.323 |
| dinov3_vits16_patch_probe | 0 | swinseg | 0.866 | - | 0.877 | 0.856 | 0.323 |
| dinov3_vits16_patch_probe | 0 | swinyseg | 0.843 | - | 0.834 | 0.852 | 0.323 |
| small_unet | 0 | all | 0.867 | 0.978 | 0.804 | 0.819 | 0.237 |
| small_unet | 0 | almeria | 0.870 | 0.980 | 0.844 | 0.786 | 0.237 |
| small_unet | 0 | shwimseg | 0.468 | - | 0.277 | 0.660 | 0.237 |
| small_unet | 0 | swimseg | 0.798 | - | 0.790 | 0.806 | 0.237 |
| small_unet | 0 | swinseg | 0.825 | - | 0.829 | 0.822 | 0.237 |
| small_unet | 0 | swinyseg | 0.820 | - | 0.811 | 0.829 | 0.237 |
