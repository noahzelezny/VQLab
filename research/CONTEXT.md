# research/: the working record, not the toolkit

Nothing here is imported by `vqlab`. For an instrument, go to the root
`CONTEXT.md`; for what is known about a model family, go to `families/`.

| path | what it is | how to treat it |
|---|---|---|
| `log/EXPERIMENTS.md`, `STATE.md`, `TIMELINE.md`, `HANDOFF.md`, `PROCESS.md`, `handoff/` | the lab notebook | **working record**: narrates attempts, including overturned ones. Never describe a released artifact from it; read its `config.json` |
| `paper/` | the paper | owned by the paper session |
| `archive/quantlab/` | the frozen quantlab-era tree (merged 2026-09-01): its scripts, shell chains, results, old model-card drafts | **history**. Scripts keep their relative paths so past chains still run; never edit them to "sync" with `src/` |
| `archive/peel/` | kernel-peel experiment notes and patches | history |

Moved out on 2026-09-25:
- the law book, to `docs/FINDINGS.md` (same section numbers);
- per-family ledgers, sweep plans and result tables, to `families/<family>/teachers/<teacher>/research/` (their sweep SCRIPTS stay in `archive/quantlab/research/<arc>/`);
- `allocation/METHOD.md`, to `docs/ALLOCATION-METHOD.md`.

## The archived quantlab `*.py` files are NOT copies of `src/vqlab/`

Twenty share a filename with a `src/vqlab/` module, and 19 of those have
diverged from it (`vq_switch.py` by about 4,100 lines). They are the versions
the archived chains ran, and past results cite their bytes. Some are frozen
on purpose and cited: `archive/quantlab/fitter_0816_cdcdeab.py` is the fitter
that produced the published 397B VQ-2.4 rung. `archive/quantlab/patch_mlx_lm.py`
installs the CURRENT runtime (`src/vqlab/runtime/vq_switch.py`), not the
archived copy.
