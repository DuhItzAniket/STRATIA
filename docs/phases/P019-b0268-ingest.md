# P019 — B0268 ingest

Status: DONE     Date: 2026-10-04 (closed 2026-10-23 with P041)     Commit: (this commit)

## Objective
Import the owner's Arducam B0268 captures with correct UTC times, location and camera metadata, from CloudScope's sky-logger sidecars or, for older frames, from EXIF.

## Inputs / dependencies
P007 registry; CloudScope P003 sidecar schema `cloudscope.sky_logger.frame/1`; `DATA/My Sample` (25 legacy frames).

## Work log
1. `stratia/data/b0268.py` — `ingest_folder()`:
   - sidecar path: schema check, SHA-256 of the image against the sidecar, UTC capture time and time source, camera read-back (exposure, gain, exposure in seconds when known), declared pointing, Sun position, clipped fraction;
   - EXIF fallback: `DateTimeOriginal` + `SubsecTimeOriginal` (local time; IST +05:30 by default), EXIF GPS; file modification time only if no EXIF time.
2. **Privacy:** the 25 Windows Camera frames carry precise GPS coordinates. STRATIA rounds every location to 0.01° (≈ 1 km), enough for Sun geometry, and never stores exact coordinates (derived tables are git-ignored anyway). The same 25 frames are public in the CloudScope repository (`legacy/B0268/`); this was reported to the owner for a decision.
3. Tests: sidecar ingest and rounding, checksum mismatch reported, EXIF local-time conversion and GPS rounding.

## Verification
- Legacy frames: 25 ingested; capture times from EXIF, 2026-09-27 10:14:21.390 → 10:19:16.842 UTC (15:44–15:49 IST, span 295 s); location rounded to (12.92, 77.51); all 4656 × 3496.
- Sun at capture (pvlib, rounded location): elevation 33.9–35.1°, azimuth 258.5–259.0°, consistent with the afternoon Sun visible in many frames.
- Unit tests: 3 passing (whole suite: 39).
- Sky-logger batch (2026-10-23, P041): 393 frames of `DATA/B0268/BLR01/2026-10-18..21` (120 / 120 / 120 / 33,
  30 s interval, 4656 × 3496) ingested from their sidecars by `scripts/build_manifest.py` `b0268()` next to the
  25 legacy frames; 418 B0268 rows in the manifest (`docs/data/manifest_summary.md`), Sun from the sidecar
  location and time; the few-shot day draws and the locked test day of P041 are built on them.

## Exit criteria
- [x] Ingests the 25 legacy frames.
- [x] Ingests a first sky-logger batch — met 2026-10-23: the four-day BLR01 batch of 2026-10-18..21 through the
      sidecar path (393 frames; P041).

## Fit & data-risk notes
- All 25 legacy frames come from one 5-minute window: they can only serve as a qualitative out-of-distribution sample, never as a test set.
- The legacy frames' labels (all "Cu", estimated by an AI agent in the prototype) are not used.

## Deviations from plan & why
Stayed PARTIAL from 2026-10-04 until the first logger batch (2026-10-18..21) was ingested on 2026-10-23 (P041).

## Next phase
P020 — External weak references (METAR).
