# Zero-shot CLIP / SigLIP 2 (P055)

Prompts from `configs/prompts.yaml` (template `a photo of the sky showing {}`), fixed before any test evaluation; image embeddings from the P051 caches; scored on the grouped val splits. Linear probes on the same embeddings are in `docs/data/baseline_probes.md` (phase P055 rows).

| Backbone | Prompt set | ccsn F1 / acc | mgcd F1 / acc | swimcat F1 / acc | montenegro F1 / acc | Mean F1 |
|---|---|---|---|---|---|---|
| clip_vitb16 | definition | 0.358 / 0.388 | 0.260 / 0.374 | 0.683 / 0.734 | 0.350 / 0.564 | 0.413 |
| clip_vitb16 | name | 0.328 / 0.349 | 0.228 / 0.320 | 0.606 / 0.696 | 0.273 / 0.491 | 0.358 |
| siglip2_b16 | definition | 0.337 / 0.357 | 0.293 / 0.421 | 0.423 / 0.506 | 0.387 / 0.589 | 0.360 |
| siglip2_b16 | name | 0.335 / 0.373 | 0.392 / 0.502 | 0.361 / 0.532 | 0.344 / 0.757 | 0.358 |
