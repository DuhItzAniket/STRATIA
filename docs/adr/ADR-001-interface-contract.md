# ADR-001 — Versioned interface contract between STRATIA and CloudScope

Status: Accepted     Date: 2026-10-04     Phase: P008

## Context
STRATIA (model) and CloudScope (system) live in separate repositories and evolve at different speeds. The model needs geometry and time context to estimate cloud-base height and to generalise across cameras, but must stay usable when a host has no calibration. Silent disagreement on a class order, sign or unit would corrupt results without any error.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Image-only model | Simplest interface | Cannot use calibration; cloud-base height and cross-camera geometry suffer |
| Raw metadata inputs (time, date, latitude, longitude, intrinsics) | Flexible | Invites shortcuts (season/location → class); every host re-implements camera models differently |
| **Image + optional per-patch ray map in a Sun-aligned frame + minimal Sun metadata, versioned contract and model card** | Physically meaningful, camera-agnostic geometry; continuous encoding; works when absent (zeros + valid flag); no climatology shortcuts | Hosts must compute ray maps (reference implementation and test vectors provided) |

## Decision
Adopt `stratia-contract` v1.0 as specified in `docs/contract.md`, with class orders defined once in `stratia/contract.py` and the model card validated by `schemas/model_card.schema.json`.

## Consequences
- Training must apply geometry dropout so the model works with zero ray maps (PLAN P050).
- CloudScope P065/P067 implement the host side; shared pixel→ray test vectors are added in STRATIA P043–P045 and CloudScope P031.
- Changes to names, shapes, channel meanings or class orders require a major version bump in both repositories.
