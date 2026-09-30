# research/: the paper, not the toolkit

Nothing here is imported by `vqlab`. For an instrument, go to the root
`CONTEXT.md`; for what is known about a model family, go to `families/`.

| path | what it is |
|---|---|
| `paper/DRAFT.md`, `paper/paper.html`, `paper/below-six-bits.pdf` | the paper: source, rendered page, PDF |
| `paper/make_results.py`, `paper/make_charts.py` | regenerate every table and figure from the saved per-position arrays (CPU; `vqlab selftest` checks this when the arrays are present) |
| `paper/regress.py` | the GPU half: the scorer still reproduces those arrays |
| `paper/RESULTS-V5.md` | the generated results tables |
| `paper/build_artifact.py`, `paper/publish/` | builds the hosted page |
