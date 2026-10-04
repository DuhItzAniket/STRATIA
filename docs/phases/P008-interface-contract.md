# P008 — Interface contract v1

Status: DONE     Date: 2026-10-04     Commit: (this commit)

## Objective
Fix the interface between STRATIA (model) and CloudScope (host): inputs, outputs, class orders, conventions, model card and versioning.

## Inputs / dependencies
CloudScope ADR-012 (time and coordinate conventions); plan overview table (`plans/README.md`).

## Work log
1. `docs/contract.md` — `stratia-contract` v1.0.
2. `stratia/contract.py` — class orders and dimensions as constants (single source of truth) plus `output_shapes()` for contract tests.
3. `schemas/model_card.schema.json` — JSON Schema for `model_card.json`.
4. ADR-001 records the decision and alternatives.
5. **Refinements over the plan draft** (also applied to `plans/README.md`):
   - `ray_map` encodes each patch's viewing direction as a **unit vector in a Sun-aligned local frame** (x toward the Sun's azimuth, z up) + valid flag, instead of raw angles: continuous (no 0/360° wrap), and zenith angle, azimuth relative to the Sun and angular distance to the Sun all follow from it.
   - `meta` reduced from 6 to 3 values (cos/sin of Sun zenith + valid): time of day, date and location are deliberately excluded to prevent climatology shortcuts.

## Verification
- Contract constants match the schema's `const` class lists, the contract document names every input and output, and output shapes are as specified (`tests/test_contract.py`, 3 tests, passing).
- CI on GitHub: P006 commit `1437c04` passed (run 37198627990).
- Conventions match CloudScope ADR-012 (azimuth from north clockwise, elevation from horizon, ENU, UTC, image origin top-left).

## Exit criteria
- [x] Contract documented and mirrored in the plan overview; CloudScope implements it in P065/P067.

## Fit & data-risk notes
Excluding time/date/location from the inputs is a deliberate guard against shortcut learning (PLAN §4).

## Deviations from plan & why
Two input refinements above; the plan overview was updated to match.

## Next phase
P009 — Research design & pre-registration.
