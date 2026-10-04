# ADR-002 — DINOv3 ViT-S/16 as the primary backbone

Status: Accepted (VRAM figures confirmed in P010)     Date: 2026-10-04     Phase: P009

## Context
Training happens on one RTX 4050 Laptop GPU (6 GB). The model needs strong dense features (sky parsing, layers) and global features (genus, CBH). Self-supervised pre-training on our own small datasets failed in the CloudScope prototype (SimCLR 39.7% on CCSN). LenghuSky-8 (CVPR 2026 Findings) showed a frozen DINOv3 ViT-L linear probe beating U-Net on all-sky segmentation. Gated access to DINOv3 ViT-S/B/L was granted on 2026-10-04.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **DINOv3 ViT-S/16 (21 M)** | Fits 6 GB for LoRA/partial fine-tuning at 512 px; strong dense features; fast feature caching | Lower capacity than B/L |
| DINOv3 ViT-B/16 (86 M) | More capacity | Fine-tuning tight on 6 GB; used as an ablation |
| DINOv3 ViT-L/16 (300 M) | Best features | Frozen extraction only on 6 GB; large caches |
| DINOv3 7B | State of the art | 26.9 GB of weights: cannot load |
| DINOv2 | Public, proven | Older; used only where DINOv3 is not needed (e.g. audits) |
| ImageNet CNNs (ConvNeXt, ResNet) | Cheap | Weaker transfer; kept as baselines |

## Decision
ViT-S/16 is the STRATIA backbone; ViT-B/16 is the main capacity ablation; ViT-L/16 is used frozen for baselines and feature studies. Loaded through `transformers` from `facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m`.

## Consequences
- Publications and redistributed weights carry the DINOv3 licence attribution ("Built with DINOv3").
- P010 measures real memory and throughput; batch sizes and resolution are set from those measurements.
