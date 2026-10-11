# Frozen-feature caches (P051)

Under `<cache_root>/features/`; built by `scripts/build_feature_cache.py`. Transform `ffr1` = full-frame resize of the P022 cached image (no crop, no mask zeroing). A fingerprint ties every cache to its row list, model and preprocessing.

| Cache | Backbone | Size | Rows | Dim | Patch tokens | Disk | Build | Fingerprint |
|---|---|---|---|---|---|---|---|---|
| `clip_vitb16_224_ffr1` | openai/clip-vit-base-patch16 | 224 | 72,974 | 512 | - | 0.17 GB | 206 s | 10887566f9b92e6e |
| `dinov2_base_224_ffr1` | facebook/dinov2-base | 224 | 72,974 | 768 | - | 0.21 GB | 221 s | d977c28e3670a92d |
| `dinov2_small_224_ffr1` | facebook/dinov2-small | 224 | 72,974 | 384 | - | 0.10 GB | 114 s | 73ddf60d576e3f77 |
| `dinov3_vitb16_224_ffr1` | facebook/dinov3-vitb16-pretrain-lvd1689m | 224 | 72,974 | 768 | - | 0.21 GB | 234 s | 953f67affe1a9d6d |
| `dinov3_vitl16_224_ffr1` | facebook/dinov3-vitl16-pretrain-lvd1689m | 224 | 72,974 | 1024 | - | 0.28 GB | 656 s | 35e9cb9f0ffe5eee |
| `dinov3_vits16_224_ffr1` | facebook/dinov3-vits16-pretrain-lvd1689m | 224 | 72,974 | 384 | - | 0.10 GB | 129 s | f5b4dfc5ef4fc9c3 |
| `dinov3_vits16_512_ffr1` | facebook/dinov3-vits16-pretrain-lvd1689m | 512 | 72,974 | 384 | - | 0.10 GB | 522 s | 9ba0acd971efe8ff |
| `dinov3_vits16_512_ffr1-patches-cbh-pool4` | facebook/dinov3-vits16-pretrain-lvd1689m | 512 | 20,942 | 384 | 64 | 0.99 GB | 184 s | 0ff51db688988954 |
| `dinov3_vits16_512_ffr1-patches-seg` | facebook/dinov3-vits16-pretrain-lvd1689m | 512 | 8,870 | 384 | 1024 | 6.51 GB | 78 s | 22860c71ff6d858e |
| `siglip2_b16_224_ffr1` | google/siglip2-base-patch16-224 | 224 | 72,974 | 768 | - | 0.21 GB | 178 s | 4cf076919c2d64ee |

Total on disk: 8.89 GB.
