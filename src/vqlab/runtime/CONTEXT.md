# runtime/ — Shipped runtime: the code that leaves this repo inside every artifact

## Inputs
Nothing. This is a library, not a stage.

## Process
`vq_switch.py` (MoE VQ layers + Metal kernels), `vq_dense.py` (dense VQLinear / VQEmbedding), `vq_pack.py` (sub-byte codes), shims (`dense_shim.py`, `glm5_shim.py`, `arch_resolve.py`), and `runtime_profile.py` (the v1.5 / v2 flag profiles).

## Outputs
Bundlers splice this text VERBATIM into `<artifact>/model.py` (see `bundle/`). Locate it with `vqlab._layout.runtime_file(name)`, never by a relative path.

## Rules that bite
- **Any edit here changes what `check-bundle` compares every published artifact against.** Run `vqlab check-bundle` across the fleet before and after.
- Dense and MoE are different runtimes (`vq_dense`'s fused path is gated on `codebook.shape[1] == 2`). A smoke test on one says nothing about the other.
- Threadgroup codebook cache: `K * dim * 2 < 32768` is a hard ceiling. Compute it FIRST.
- Don't rename or reformat these files, and don't edit their `vqlab.*` import strings: that text is what published bundles carry.
- **SKIPZERO switch** (docs/SKIPZERO.md): `VQSwitchLinear(row_table=...)` serves compact live rows; `#if SZ` in the d4 WALK decode kernel and gemmseg2 (both arms) is the ONLY kernel code involved, compiled only when a row table is present (spec kernels, `_sz` name suffix). Any other dispatch with a row table REFUSES. Non-sz kernels are byte-identical to before (`tests/test_runtime_equivalence.py`). `skipzero_weights()` converts the `sz-pack` tensors at load, lazily.
- **STALE vs FAIL.** `check-bundle` reports STALE (exit 0; 2 with `--strict`) when a bundle carries an earlier runtime listed in `equivalent_revisions.json`: byte-identical OUTPUT, only missing speedups. An entry goes in ONLY with `tests/test_runtime_equivalence.py` passing against it (every fleet geometry, decode + prefill N, fp16 + bf16). A kernel change that is not bit-exact never gets an entry: that is an instrument change (KL gate), and old bundles stay FAIL.

