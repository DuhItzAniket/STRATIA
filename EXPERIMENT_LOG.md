# Experiment Log

One entry per experiment run that informs a decision. The machine-readable record is `runs/registry.csv` (git SHA, config hash, data hash, metrics); this log adds the human reasoning.

| Date | Run ID | Phase | Question | Result (val unless stated) | Decision |
|---|---|---|---|---|---|
| 2026-10-04 | 20261004T112119Z_repro_check_a/b_b011a8395a | P004 | Are two runs with one config bit-identical on this GPU? | Yes: identical loss/accuracy for all 3 epochs (acc 0.5688 / 0.86945 / 0.91775) | Deterministic mode is adequate; still report mean ± std over seeds for real models |
