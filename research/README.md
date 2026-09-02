# research/ — active arcs

New research lives here, one directory per arc, LEDGER.md as the lab
notebook (same conventions as ever: every claim cites its measurement,
negative results are banked, the 3x-noise-floor bar for calling a
single-fit result).

**The old research is still the reference.** The full quantlab history —
the fits, sweeps, refuted hypotheses and build provenance for every
published artifact — is preserved under `research/quantlab/` (603
commits of history merged 2026-09-01). Check there FIRST before re-deriving
anything about an existing artifact; the answer is usually already
measured.

Active arcs:
- `flash-next-recipe/` — dense-precision rebalance (opened 2026-09-02:
  ablation showed 56% of the 2.1bpw decode gap is 8-bit dense reads,
  not VQ kernels; see research/quantlab/research/flash-next/LEDGER.md).
