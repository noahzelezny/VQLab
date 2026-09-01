# Two VQ runtimes, one silently shadowing the other

**How three published artifacts shipped broken while every check passed.
Written 2026-09-01 after the dense 27B incident; the mechanism is still
live for MoE artifacts and is a decision, not a bug to patch.**

## There are two delivery mechanisms, not one

**The bundle.** `model.py` inside the checkpoint, loaded by stock mlx-lm via
`spec_from_file_location("custom_model", ...)` when `config.json` names
`model_file`. This is what the model cards describe and what a downloader
gets: *"Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint."*

**The patch.** `quantlab/patch_mlx_lm.py` copies `vq_switch.py` into
`site-packages/mlx_lm/models/` and inserts a hook into `load_model`.

Both exist. Only the first is documented on the cards.

## The patch silently wins

The hook runs immediately before `model.load_weights` and does this:

```python
_vq_prefixes = sorted({k[:-6] for k in weights if k.endswith(".codes")
                       and weights[k].ndim == 3})
if _vq_prefixes:
    from .models.vq_switch import VQSwitchLinear
    for _p in _vq_prefixes:
        setattr(_obj, _parts[-1], VQSwitchLinear.from_weights(...))
```

It **overwrites the modules the bundle just created** with the copy from
site-packages. So on any patched machine, an expert-shaped artifact's own
runtime is inert: the bundle is loaded, then discarded.

This had already bitten once. The hook's own comment records it: *"Dense VQ
artifacts install their own modules via model.py; this hook overwriting them
with the expert-shaped VQSwitchLinear breaks them (2026-08-19)."* The fix
narrowed the hook to `ndim == 3` rather than removing the shadowing, so the
same mechanism kept operating for every MoE artifact.

## Why that turned into shipped breakage

1. **Every development machine is patched.** Both build venvs
   (`~/vqvenv`, `~/qlab-venv`) carry `mlx_lm/models/vq_switch.py`. So our
   machines were unrepresentative in precisely the dimension that mattered:
   the bundle could be broken and nothing here would notice, because nothing
   here was running it.
2. **Nothing compared the two.** `check-bundle` checked substrings.
   `vqlab smoke` printed which runtime resolved and asserted nothing — its
   own docstring said the resolved copy "depends on the loader and the
   environment" and then only printed it.
3. **So they drifted, invisibly.** exo's installed copy is **228 lines, dated
   Aug 15 (M4) / Aug 24 (M3)**: `d=4` only, `_fused()` with no `pack_bits`
   parameter, no d8 kernels. The bundle is **2048 lines** with d8, packed and
   tiled kernels. Two weeks apart, and no check would ever say so.
4. **The dense bundle broke and only downloaders saw it.** `model.py`
   imported `mlx_lm.models.vq_switch` — which resolves on every patched
   machine and on none of theirs.

## Consequences that are still live

- **exo does not run the bundle.** `utils_mlx.py` calls
  `load_model(model_path, lazy=True, strict=False)`, and that signature is
  `trust_remote_code: bool = False`, so the bundle is not even read; the
  patched runtime supplies the modules. None of the kernel work from the last
  two weeks reaches exo-served inference.
- **exo's runtime cannot serve the current MoE artifacts.** They are
  `d=8 / K=16384 / pack_bits=14`; the installed copy has no packing support
  and no d8 kernels at all.
- **The name collision follows from this.**
  `~/.exo/models/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw` on the M4 is an
  old `K=128 / d=4` fit. That is not a stray copy: it is the generation the
  Aug-15 runtime was built for. An old model and an old runtime that match
  each other, under the current name.
- **A card claim needs re-verifying.** The 397B card says exo serving was
  *"verified ... with an unpatched `mlx-lm`, producing output identical to the
  patched run."* That may well have been true when written; it cannot be
  demonstrated on these machines now, because both exo venvs are patched.

## What is already fixed

`vqlab smoke --strict` (2026-09-01) fails when the runtime resolves from
anywhere a downloader would not have it, and **treats the presence of
`mlx_lm.models.vq_switch` as a failure in its own right** — that check exists
precisely because "inside site-packages" was never the right test. `vqlab
publish` runs the gate before upload with no override. Together those close
the "our machines are unrepresentative" hole.

## What is not decided

**Whether exo should use the bundle or the patch.** Having both is the
problem, not either one:

- *Bundle only* matches the model cards and gives one artifact one runtime.
  It needs exo to pass `trust_remote_code=True` into `load_model` (a change
  to the fork, alongside the existing 8-line codebook-replicate guard), and
  the patch removed from every venv.
- *Patch only* keeps exo's current behaviour and requires re-running
  `patch_mlx_lm.py` on every venv whenever the runtime changes, plus a gate
  that compares the installed copy against the repo. The cards would need to
  stop saying exo runs stock mlx-lm.

Either is defensible. What is not defensible is the present state, where the
cards describe one mechanism, the machines run the other, and nothing
compares them.
