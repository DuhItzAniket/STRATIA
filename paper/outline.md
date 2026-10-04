# Paper outline (working)

Working title: *STRATIA: Single-Camera Sky Understanding with Calibrated Cloud-Base Height*
(alternatives in `docs/PLAN.md`; final title chosen after Gate G4)

1. **Introduction** — sky cameras are cheap; leaderboards overstate recognition (leakage, expert disagreement); geometry foundation models ignore the sky; contributions.
2. **Related work** — ground-based cloud classification and its protocols; cloud segmentation incl. multi-layer; cloud-base height from camera networks and ceilometers; Cloud4D and cloud tomography; vision foundation models and the sky.
3. **Data and protocol** — harmonised WMO hierarchy, set-valued labels, leakage audit, LODO, expert-disagreement evaluation, ceilometer pairing.
4. **Method** — DINOv3 backbone, Sun-aligned ray-map encoding with geometry dropout, heads (genus/étage, oktas, sky parsing, layers, ordinal CBH, reliability), losses, training stages.
5. **Experiments** — baselines; recognition (in-domain, LODO, Montenegro ceiling); cloud-base height (CDLRA, held-out CDLRB, B0268 qualitative); calibration; ablations; efficiency.
6. **Limitations** — single view ambiguity, ceilometer point sampling, dataset biases, B0268 labels.
7. **Conclusion**

Every table and figure is generated from results files (`docs/PLAN.md` P098); this outline is filled in as phases complete.
