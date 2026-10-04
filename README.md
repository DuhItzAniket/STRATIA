# STRATIA

**STRATIA — Spatio-Temporal Representation for Atmospheric Intelligence and Analysis.**
A single-camera sky-understanding model: from one ground-based sky image (plus optional time, location and calibration), STRATIA predicts cloud genus, cloud layers (low / mid / high), cloud cover, cloud-base height with calibrated uncertainty, a sky / cloud / sun-glare / obstruction map, and a reliability score.

> **Status:** Stage A (foundation) in progress. See [`PROJECT_STATE.md`](PROJECT_STATE.md) and the plan in [`docs/PLAN.md`](docs/PLAN.md).

STRATIA is **only the model**. It never controls hardware. The sky-observation system that captures images, controls the camera mount and runs STRATIA in the field is [CloudScope](https://github.com/DuhItzAniket/CloudScope). The two projects share a versioned interface contract (`docs/contract.md`, from phase P008).

## Repository layout

| Path | Contents |
|---|---|
| `stratia/` | Python package: data, geometry, models, losses, training, evaluation, export |
| `configs/` | Experiment and dataset configuration |
| `scripts/` | Command-line entry points (inventory, audits, training, evaluation, export) |
| `tests/` | Unit, contract and sanity tests |
| `docs/PLAN.md` | The 100-phase research and training plan |
| `docs/phases/` | One document per completed phase |
| `docs/adr/` | Architecture decision records |
| `docs/data/` | Data cards and audit reports |
| `paper/` | Paper sources; tables and figures generated from results |

Datasets, runs, caches and model weights are never committed; datasets are referenced by manifest and SHA-256.

## Development process

Every phase ends with a phase document, a commit `P###: <title>` carrying a `Phase-Status:` trailer, and a push. Failed or partial results are documented, not hidden; no number in a document or the paper may lack a traceable run.

## Licence

Code: Apache-2.0 (`LICENSE`). Datasets keep their own licences (recorded per dataset); model weights carry the licences of what they were built from (e.g. the DINOv3 licence requires attribution).
