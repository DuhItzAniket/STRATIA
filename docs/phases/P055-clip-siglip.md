# P055 — B4: CLIP / SigLIP 2

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
Language-supervised baselines: zero-shot recognition with CLIP ViT-B/16 and SigLIP 2 B/16 under two fixed prompt
sets (class names vs WMO-style definitions), and linear probes on the same embeddings, so the paper can say what a
contrastive backbone knows about clouds before any supervision.

## Inputs / dependencies
P051 caches (`clip_vitb16_224`, `siglip2_b16_224`, with the text towers in `stratia/features/backbones.py`); P054 probe
code and tables; P034 native classes.

## Work log
1. `configs/prompts.yaml`: for every native class of CCSN, MGCD, SWIMCAT and Montenegro a `name` ("cirrus clouds")
   and a `definition` (one line in the International Cloud Atlas's wording), template "a photo of the sky showing
   {}". Written once, before any evaluation, and not edited after seeing a number (the plan's guard against prompt
   overfitting; test stays locked anyway).
2. `scripts/baseline_zeroshot.py`: text embeddings per prompt set, cosine against the cached image embedding
   (both L2-normalised, the model's logit scale and SigLIP's bias applied), argmax over the source's classes, macro-F1
   and accuracy on the grouped val split; 16 rows in `data/baseline_zeroshot.parquet`, table in
   `docs/data/baseline_zeroshot.md`.
3. Linear and k-NN probes on the two embeddings through `scripts/baseline_probes.py --phase P055` (rows in
   `docs/data/baseline_probes.md`).
4. Fix on the way: transformers 5 returns the text tower's output object from `get_text_features`; the projected
   embedding is its `pooler_output` (`text_features` now unwraps it; the first run crashed on `.float()`).

## Verification
- Zero-shot, macro-F1 on val (chance: CCSN 0.09, MGCD 0.14, SWIMCAT and Montenegro 0.20):

| Backbone | Prompts | CCSN | MGCD | SWIMCAT | Montenegro | Mean |
|---|---|---|---|---|---|---|
| CLIP ViT-B/16 | definition | 0.358 | 0.260 | 0.683 | 0.350 | **0.413** |
| CLIP ViT-B/16 | name | 0.328 | 0.228 | 0.606 | 0.273 | 0.358 |
| SigLIP 2 B/16 | definition | 0.337 | 0.293 | 0.423 | 0.387 | 0.360 |
| SigLIP 2 B/16 | name | 0.335 | 0.392 | 0.361 | 0.344 | 0.358 |

- Linear probe, macro-F1 on val: CLIP 0.597 / 0.758 / 1.000 / 0.627 (mean 0.746, LODO étage mean 0.371); SigLIP 2
  0.594 / 0.762 / 1.000 / 0.614 (mean 0.743, LODO 0.529). On the mean both sit with DINOv2-B (0.738) and below
  DINOv3-B (0.765); CLIP is the weakest LODO backbone.
- Reading: definitions help CLIP on all four sources (+0.055 mean); for SigLIP 2 the two prompt sets tie on the mean
  and swap places by source. Zero-shot is 2–4× chance but 0.3–0.4 below a linear probe on the same features: the
  genus vocabulary is in the text tower only loosely. Montenegro's altitude classes (not genera) are the hardest for
  prompts and the easiest for accuracy (0.757: the majority class dominates).

## Exit criteria
- [x] Table (zero-shot and probes).

## Fit & data-risk notes
- Prompt guard honoured: two prompt sets, one pass, no iteration against val. Any later prompt change is a new
  prompt-set key and is reported next to these, not in their place.
- The SigLIP 2 tokenizer needs `padding="max_length"` (64) as trained; CLIP uses 77 with truncation.

## Deviations from plan & why
- None.

## Next phase
P056 — B5: small VLM (deferrable; a sanity check).
