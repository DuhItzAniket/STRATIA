# P098 — Paper draft

Status: DONE (full draft; not compiled locally)     Date: 2026-11-05     Commit: (this commit)

## Objective
Plan (Stage L, P098): CVPR template; all tables and figures generated from the results JSON (no hand-typed
numbers); full draft by 6 Nov.

## Inputs / dependencies
- The CVPR 2027 author kit (`paper/cvpr.sty`, `paper/preamble.tex`, `paper/ieeenat_fullname.bst`, fetched from
  github.com/cvpr-org/author-kit); `paper/outline.md`; `scripts/reproduce_tables.py` (P097) for
  `paper/tables/*.tex` and `paper/numbers.tex`; `scripts/paper_figures.py` for `paper/fig/*.png`; the P090–P094
  numbers; the citation checks of P099.

## Work log
1. `paper/main.tex` (review mode, anonymous, `\confName{CVPR}`, `\confYear{2027}`), sections in `paper/sec/`:
   abstract, introduction (three problems: leakage, forced labels, one camera; four contributions), related work,
   data and protocol (sources, leakage audit, harmonisation, splits and the lock), method (backbone and SkyRay,
   heads and losses, training, calibration, adaptation), experiments (criteria, recognition, human ceiling,
   cloud-base height, sky parse / OOD / glare, ablations, efficiency, the owner's camera), limitations
   (both failed criteria, geography, camera identity, labels, what is not measured), conclusion. 4,043 words of
   prose plus seven tables and three figures, within the eight-page CVPR limit by estimate (to be confirmed at
   compile time).
2. Every number in the prose is a macro from `paper/numbers.tex` (43, generated) or a value that appears in a
   generated table; `tests/test_paper.py` fails if a headline value is typed into a section, if a cite key, label,
   macro, input or graphic does not resolve, or if braces are unbalanced.
3. Figures (`scripts/paper_figures.py`): the architecture block diagram (Fig. 1), the two P093 confusion heatmaps
   side by side, and the CBH validation scatter with 80 % conformal intervals (1,368 cloudy pairs, coverage 0.84).
4. Bibliography `paper/main.bib`: 18 entries, verified against Zenodo / Crossref / arXiv / PMLR on 2026-11-05
   (P099); the three dataset records cite their Zenodo DOIs.
5. Two pre-registered negatives are in the abstract and in Table 1 with the same intervals as the positives; the
   two evaluations of P090 are stated in the data section ("read once, with the reason logged").

## Deviations from the plan
- Compiled on 2026-11-06 with MiKTeX (installed with the owner's permission): `pdflatex main; bibtex main;
  pdflatex main; pdflatex main` gives 8 pages including the references, no overfull boxes, no undefined references
  or citations, after two layout fixes (the wide tables as `table*`, the efficiency table's labels shortened).
  `supp.tex` compiles to 22 pages clean. The PDFs are build products (not committed); any machine with TeX
  reproduces them from `paper/`.
- The Montenegro test interval is blank in Table 1 (two blocks of days; the bootstrap is undefined), as recorded at
  P090.

## Outputs
- `paper/main.tex`, `paper/sec/*.tex`, `paper/main.bib`, `paper/fig/*.png`, `paper/tables/*.tex`,
  `paper/numbers.tex`, `scripts/paper_figures.py`, `tests/test_paper.py`.
