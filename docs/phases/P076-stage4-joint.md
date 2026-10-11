# P076 — Stage-4: joint multi-task (frozen)

Status: DONE     Date: 2026-10-11     Commit: (this commit)

## Objective
All heads of the model trained together on cached features with the P069 sampler, the P070 weighting and the
gradient-conflict monitor; compared head by head with the single-task runs of P073–P075, so negative transfer is
measured rather than assumed.

## Inputs / dependencies
P073 global rows (with Eye2Sky étage), P074 dense data, P075 CBH data (`stratia/train/{data,dense_data,cbh_data,
global_eval}.py`); P071 `STRATIA` model without a backbone; P072 engine.

## Work log
1. `scripts/train_joint.py`: three batch groups (global 64 rows on the 8 × 8 grid, dense 16 images on the 32 × 32
   grid, CBH 64 frames on the 8 × 8 grid with ray maps), group probabilities 0.5 / 0.3 / 0.2, Kendall weighting over
   genus / étage / oktas / sky / layer / cbh, the hierarchy term on global batches and the CBH–étage consistency
   term on CBH batches (through the global heads on the same tokens), task-gradient cosines on the shared
   parameters (geometry encoder + global heads) every 100 steps from the last batch of each group; early stopping
   on the mean of étage macro-F1, dense mIoU and CBH skill. Evaluation = the P073 / P074 / P075 scorers on the
   same val sets; `docs/data/stage4_joint.md` lists every head's joint vs single-task number.
2. Runs: `joint` (400 batches / epoch, 15 epochs, 11 min) and `joint_long` (1,000 batches / epoch, stopped at
   epoch 12, best 7, 23 min).

## Verification
Validation, best epoch of each run; single-task = P073 `indomain_e2s`, P074 (Kendall), P075 base:

| Head / metric | Single-task | Joint (400 / epoch) | Joint (1,000 / epoch) |
|---|---|---|---|
| Global: mean étage macro-F1 | **0.817** | 0.812 | 0.799 |
| Dense: sky mIoU (all) | **0.905** | 0.870 | 0.881 |
| Dense: Almería layer mIoU | **0.559** | 0.544 | 0.421 |
| CBH: MAE (m) / skill | 566 / +0.506 | 618 / +0.461 | **545 / +0.525** |
| CBH: 80 % coverage | 0.92 | 0.94 | 0.93 |

- Task-gradient cosines on the shared parameters (113–119 logged steps of the long run): every pair averages
  within ±0.04 of zero; the conflicts are episodic: sky vs layer is below −0.1 in 39 % of the logged steps
  (minimum −0.92), CBH vs layer in 16 % (minimum −0.29); all other pairs below −0.1 in ≤ 5 % of the steps.
- Reading: joint training is **better for the CBH head** (545 m, 21 m below the single-task base, twice the seed
  std) once it gets enough steps, **neutral to slightly worse for the global heads** (−0.018 étage), **worse for
  the dense heads** (−0.024 sky mIoU) and **clearly worse for the layer head** (−0.14), the head whose gradients
  conflict most often with sky parsing. The plan's exit criterion ("joint ≥ single on most heads, else documented")
  is met in the second form: documented, with the gradient monitor naming the pair.

## Exit criteria
- [x] Joint vs single-task per head, with the negative transfer documented and attributed (sky ↔ layer, CBH ↔ layer).

## Fit & data-risk notes
- The dense heads get 30 % of the batches in the joint run against all of them in P074 (1,800 vs 5,800 dense
  batches at 400 / epoch; 4,500 at 1,000 / epoch); part of the sky-parsing gap is budget, and the long run closes
  0.011 of it. The layer head's loss is not budget: it falls with more joint training.
- Decision for P077 / P081: the joint checkpoint initialises the LoRA stage (it carries every head), and the fit
  sign-off reports both the joint and the single-task candidates; the paper's dense numbers come from the
  single-task dense run unless Stage I changes the picture.

## Deviations from plan & why
- Two joint runs instead of one, to separate "undertrained" from "conflicting" (the long run answers it).

## Next phase
P077 — Stage-5: partial unfreeze.
