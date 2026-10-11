# P056 — B5: small VLM

Status: PARTIAL     Date: 2026-10-11     Commit: (this commit)

## Objective
A sanity check, as the plan marks it (deferrable): what a small open vision-language model answers when asked to
name the cloud type from a closed list, so the paper can place "ask a VLM" against the probes and zero-shot CLIP
without claiming a tuned baseline.

## Inputs / dependencies
P022 image cache; P038 grouped val splits; `configs/prompts.yaml` class names (P055); `Qwen/Qwen3-VL-2B-Instruct`.

## Work log
1. `scripts/baseline_vlm.py`: 100 random grouped-val images per source (seed 0) for CCSN and MGCD, 448 px, the
   chat prompt lists the source's class names and asks for one of them, greedy decoding, the answer is parsed by the
   first class name it contains (unparsed answers count as wrong and are reported). fp16 on the GPU: bitsandbytes has
   no Windows wheel in the lock, so the plan's 4-bit variant did not run. One pass, no prompt tuning.
2. One registry run (phase P056) with the per-source numbers as summary; rows in `data/baseline_vlm.parquet`, table
   in `docs/data/baseline_vlm.md`. 8 minutes for 200 images.

## Verification
| Source | n | Macro-F1 | Accuracy | Unparsed |
|---|---|---|---|---|
| CCSN (11 genera) | 100 | 0.225 | 0.250 | 0.01 |
| MGCD (7 types) | 100 | 0.369 | 0.400 | 0.00 |

- Against P055 on the same sources: CLIP zero-shot with definition prompts 0.358 (CCSN) / 0.260 (MGCD); SigLIP 2
  with names 0.335 / 0.392. The 2B VLM is below CLIP on the genera and level with SigLIP 2 on MGCD's coarser types;
  all three are far below a linear probe (0.58–0.65 / 0.76–0.83). With 100 images the interval on each number is
  about ±0.09.

## Exit criteria
- [x] Table (sanity check).

## Fit & data-risk notes
- The answers were parsed, not scored by likelihood: a model that hedges between two names is scored on the first.
  The unparsed rate (≤ 1 %) says the closed list was followed.
- Not a tuned baseline: no prompt search, no few-shot, no 4-bit; a larger VLM or a tuned prompt could move these
  numbers, which is why they are not used as a reference by any criterion.

## Deviations from plan & why
- fp16 instead of 4-bit (no bitsandbytes on Windows); 200 images, not the full val sets (8 minutes per 200 at
  448 px on the 6 GB card; the full sets would take 1.5 hours for a sanity check). PARTIAL on both counts.

## Next phase
P057 — B6: segmentation baselines.
