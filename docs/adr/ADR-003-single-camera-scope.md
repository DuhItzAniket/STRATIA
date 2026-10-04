# ADR-003 — Single-camera scope for STRATIA v1

Status: Accepted     Date: 2026-10-04     Phase: P009

## Context
Multi-camera stereo networks already measure cloud-base height well (Blum et al. 2021 on the Oldenburg stations; a 2026 Valladolid network with R² 0.93), and Cloud4D (NeurIPS 2025) reconstructs 4D cloud fields from camera pairs with four H100 GPUs. A single low-cost camera is the common case (including the owner's Arducam B0268) and has no published learned cloud-base-height method. The owner wants STRATIA to work from single-camera data and to train on one laptop GPU.

## Decision
STRATIA v1 takes **one image from one camera** (plus optional calibration and Sun position). Multi-camera information may be used **only as training-time supervision** in future versions (e.g. stereo pseudo-labels), never as an input. Ceilometer measurements supervise cloud-base height directly.

## Consequences
- Cloud-base height is defined at the zenith (or optical axis) of the single view; per-pixel height maps are future work.
- Expected accuracy is bounded: claims are framed as skill over climatology and calibrated intervals, not stereo-level accuracy.
- The comparison set includes climatology, a ceilometer-supervised CNN, genus-to-étage lookup, and zero-shot geometry foundation models.
