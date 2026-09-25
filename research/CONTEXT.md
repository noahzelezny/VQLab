# research/: working record, not the toolkit

Nothing here is imported by `vqlab`, and nothing here is maintained as a tool.
For an instrument, go to the root `CONTEXT.md` and run `vqlab <cmd>`.

| path | what it is | how to treat it |
|---|---|---|
| `quantlab/FINDINGS.md` | the law book: settled laws, retracted leads, instrument and Metal rules | **reference**: read it before proposing any quantization idea |
| `quantlab/EXPERIMENTS.md`, `STATE.md`, `TIMELINE.md`, `HANDOFF.md` | lab notebook | **working record**: narrates attempts, including overturned ones. Never describe a released artifact from it |
| `quantlab/MODEL_CARD_*.md` | card drafts from the quantlab era | superseded by `docs/model-cards/` |
| `quantlab/paper/` | the paper | owned by the paper session |
| `quantlab/*.py`, `*.sh` | **frozen quantlab-era tools and the chains that ran them** (last synced 2026-08-22) | see below |
| `quantlab/research/<family>/` | per-family ledgers and sweep plans | reference for that family |
| `flash-next-recipe/`, `peel/` | family recipe / kernel-peel notes | reference |

## The quantlab `*.py` files are NOT copies of `src/vqlab/`

Twenty of them share a filename with a `src/vqlab/` module, and 19 of those
20 have diverged from it: `vq_switch.py` by ~4,100 lines, `check_release.py`
by ~440. They are the quantlab-era versions the `*.sh` chains in this folder
ran (17 chains call them by relative path). Past results that cite them were
produced by THESE bytes. Keep them frozen and never edit them to "sync";
fix the `src/vqlab/` version instead. Some are frozen on purpose and cited:
`fitter_0816_cdcdeab.py` is the fitter that produced the published 397B
VQ-2.4 rung.

**Hazard:** `quantlab/patch_mlx_lm.py` installs `quantlab/vq_switch.py` (the
2026-08-22 runtime) into an mlx_lm tree. Re-running it today installs a stale
runtime. The current runtime is `src/vqlab/runtime/vq_switch.py`
(`vqlab._layout.runtime_file("vq_switch.py")`).
