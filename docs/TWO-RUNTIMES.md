# The VQ runtime: which copy actually runs, and how we kept getting it wrong

**Written 2026-09-01 after the dense 27B incident. The conclusions below were
wrong twice before they were right, both times for the same reason, and that
reason is the point of the document.**

## Current state (verified, three sessions independently)

**The live exo serving path is clean and runs the artifact's own bundle.**

| | mlx-lm | `load_model` hook | `models/vq_switch.py` |
|---|---|---|---|
| `/opt/anaconda3/envs/exo` (M3, **live**) | 0.31.9 | **none** | **absent** |
| `/opt/homebrew/anaconda3/envs/exo` (M4, **live**) | 0.31.9 | **none** | **absent** |
| `~/exo/.venv` (both boxes, **dormant**) | 0.31.9 | present, line 501 | present, 228 lines |

The running process on the M3 is `/opt/anaconda3/envs/exo/bin/python3.13`;
the M4's supervised process is the equivalent homebrew conda env. mlx-lm
0.31.9's `load_model` takes no `trust_remote_code` argument at all and
executes a config's `model_file` unconditionally, so on the live path the
bundle loads and **nothing overwrites it**.

So a re-bundled artifact does reach exo-served inference, kernels and all.
The model cards' claim that exo runs stock `mlx-lm` is true of every env that
can actually serve.

## Two real issues remain

**A dormant landmine.** `~/exo/.venv` carries the `patch_mlx_lm.py` hook
inside `load_model` (line 501) plus a 228-line `vq_switch.py` from Aug 15/24:
`d=4 only, v1`, `VQSwitchLinear.from_weights(codes, codebook, vq_scales)`
with no `pack_bits`. Launch exo from the fork's own venv and that hook
overwrites the bundle's modules for every `ndim == 3` `.codes` tensor, then
hands a `pack_bits=14` artifact's packed uint32 words to a runtime that has
no packing support. Expect a shape error or silent garbage, presenting as a
bug in the artifact rather than in the environment. Deleting the hook and
the module from both dormant venvs removes it.

**A forward-compat trap.** mlx-lm >= 0.32 adds `trust_remote_code` to
`load_model` and defaults it to `False`. exo calls
`load_model(model_path, lazy=True, strict=False)` and never passes it, and
`model_file` appears nowhere in exo's source — so a routine pin bump silently
turns every VQ artifact into a load-time `ValueError`. The fix is to thread
`model_card.trust_remote_code` through both call sites in
`utils_mlx.py` (detecting the kwarg with `inspect`, so it no-ops on 0.31.9).
Drafted in the fork; not committed.

Note the asymmetry: the dormant hook is a landmine that only fires if someone
picks the wrong binary, while the pin bump fires on an ordinary dependency
update and takes everything with it.

## How this document was wrong twice

Both errors were the same error, and it is the same one that shipped three
broken artifacts.

**First: "exo never runs the bundle."** I read `load_model`'s signature from
`~/.venvs/qwen4exp` — a 0.32 install — and applied it to exo's call sites.
exo pins 0.31.9, where the argument does not exist. I inspected the wrong
environment and reported it as fact.

**Second: "the hook is live on both boxes."** I then measured `~/exo/.venv`,
found the hook and a two-week-stale runtime, and concluded exo's serving path
was shadowed. The hook is really there and really stale — but that venv is
dormant. exo is launched from a conda env that has neither. I inspected the
wrong environment again, this time more carefully, and was wrong again.

The peer session had the complementary halves each time: right about the pin
and wrong about the hook's presence, because it had grepped the conda envs.
Neither of us was careless; we were each looking at a real environment and
assuming it was THE environment.

**That is precisely the original defect.** Three artifacts shipped a
`model.py` importing `mlx_lm.models.vq_switch` because it resolved in the
venvs we built and tested in. The gate now refuses to certify from an
environment a downloader would not have. This document then reproduced the
same mistake twice while explaining it.

The rule worth keeping: **before reporting what an environment does,
establish that it is the one that runs.** `ps aux` on the live process beats
any amount of reading in a plausible-looking directory. Every claim in the
table above is backed by the running binary, not by a path that looked right.

## What is not decided

Whether to delete the hook and `vq_switch.py` from the dormant `~/exo/.venv`
on both boxes (recommended, removes the landmine, costs nothing since that
venv serves nothing), and whether to land the `inspect`-based
`trust_remote_code` change in the fork before any pin bump (recommended, and
the more urgent of the two). Both are Noah's; no venv has been modified.
