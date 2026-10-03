# skipzero/ — vq-skipzero: drop dead VQ rows on disk (a shipped format since 2026-09-29)

**This is an experiment, not a shipped format.** Nothing here is in `runtime/`,
no published artifact uses it, and the shipped vq_switch runtime and kernels
are untouched. Stage 1 saves DISK / download bytes only; RAM and decode
bandwidth are unchanged because the rows are re-expanded at load.

## Inputs
A VQ MoE artifact (config `vq_modules`, packed or unpacked codes). The finding
it acts on: `vqlab zero-groups` -- the 397B rungs spend ~10% of text bytes on
groups with vq_scales < 6.2e-5, ~98% of them in FULLY-dead [expert, out_row]
rows, concentrated in L0-L4.

## Process
- `vqlab sz-pack <artifact> --dry-run [--layers A-B]` prices it from vq_scales
  alone (seconds, writes nothing).
- `vqlab sz-pack <artifact> --out <scratch>/<new> [--layers A-B]`
  writes a NEW dir: rewritten shards hold `{p}.sz_shape / sz_rowmask / sz_codes /
  sz_scales` (live rows only, code words and scales byte-identical); shards with
  no dead rows and every other file are symlinked; config.json gains a
  `vq_skipzero` block; model.py = the source bytes + an appended hook;
  `skipzero_load.py` (the expansion shim) is copied in; a build record is written.
- `vqlab sz-check <packed> <original>` (numpy, no GPU): live rows byte-identical,
  dead rows expand to code 0 / scale 0, every passthrough tensor identical.
  `vqlab sz-check --selftest` runs a synthetic pack + numpy AND mlx-CPU expansion.

## Outputs
A vq-skipzero artifact under vqlab-scratch/ and a printed size report.

## Rules that bite
- **Not bit-exact.** Dead rows become EXACT zeros where the source had ~1e-5 x
  codebook values. It needs a paired `kl-ladder` against the source before any
  claim (on the 397B: `--lazy-over-gb 16`, the full-vocab caches in
  `families/qwen3_5/teachers/Qwen--Qwen3.5-397B-A17B-bf16/teacher_caches.json`).
- **The hook is load_weights(), not only sanitize().** mlx_vlm SKIPS sanitize for
  `format=mlx` shards (every artifact here); both loaders call load_weights.
  `expand_weights` is idempotent so both hooks may run.
- **model.py drifts from the repo runtime by design** (appended hook):
  check-bundle will report drift. Pin + smoke the packed copy before measuring.
- **Disk.** A full 397B pack rewrites ~27 of 29 shards (~96 GiB new). Dry-run
  first; `sz-pack` refuses when free space < new bytes + 5 GiB.
- The 35B barely benefits (0.6%): the finding is a 397B-teacher property.
- Never write outside vqlab-scratch/; never overwrite (`--out` must not exist).

## Stage 2: compact RESIDENT (EXPERIMENTAL, not gated yet)
Keeps live rows' codes/scales + an int32 [E, OUT] row table (-1 = dead)
resident; dead rows read no code bytes and output exact zeros.
- `sz_resident.py` -- copied into the artifact; `install(ns)` detects the
  runtime VINTAGE from its kernel text and forks exactly two kernels by
  EXACT-TEXT patch (each asserted to match once); anything else refuses:
  * `walk` (repo runtime, 35B): `_SRC_FUSED_PACKED_D4_WALK` ->
    `sz_fused_packed{B}_d4_walk`, `_SRC_GEMMSEG2` -> `sz_gemmseg2_packed{B}_d4...`.
    Packed uint32 d4 only.
  * `u8` (397B bundle, profile v1.5): `_SRC_FUSED_PACKED` -> `sz_fused_packed8`
    (compact uint8 [NLIVE, NSUB] rows viewed as uint32, the bundle's U8-VIEW),
    `_SRC_GEMMSEG2` (BITS=0, CT=uchar) -> `sz_gemmseg2_u8_d4...`. Unpacked
    uint8 d4 K<=256, NSUB % 4 == 0 only.
  The shipped runtime is untouched. Dead-row arithmetic mirrors the expanded
  path in both (decode +0; prefill stages `(half)(0.0f * cb[0])`), so the
  target is BYTE equality with stage 1.
- `vqlab sz-resident <stage1> --out <scratch>` -- new dir: symlinked shards,
  config `vq_skipzero.resident=true` + `runtime_vintage` + per-module
  code_words/code_dtype/scale_groups, model.py = source bytes +
  `sz_resident.MODEL_HOOK`. Refuses a source model.py whose kernel text is
  not a forked vintage or whose vintage does not match the codes dtype.
  Stage-1 artifacts are unaffected (the flag is opt-in).
- `vqlab sz-bitexact <resident> <reference> [--ref2 <stage1>] [--mem-ref <stage1>] --out <scratch>`
  -- one process per artifact, compares module outputs (N=1/8/4096/4097),
  in-model module outputs, logits, a 9k chunked prefill and a 32-token greedy
  run AS BYTES. `--synthetic [--vintage walk|u8|both]` = GPU many-dead-rows
  stress vs EACH vintage's own kernels (u8: the 397B bundle's model.py text,
  exec'd read-only up to its arch shim). `--partial A-B <resident> <stage1>
  [--original <rung>]` = real switch modules of layers A-B, no full model
  (for the M3, which cannot hold the 397B). `--selftest` = CPU only.
- Rules: the row table costs 4 B/row. On the 35B the net resident saving is
  only ~0.05 GiB. On the 397B-2.4 it is 12.31 GiB of rows - 0.375 GiB of
  tables = 11.94 GiB net. The 18 down_proj modules save 0.021 GiB of rows but
  cost 0.14 GiB of tables, so they are a net LOSS.
- The 397B bundle has no `_reach_vq`; the hook passes `globals().get("_reach_vq")`
  and swap_modules falls back to `sz_resident.reach` (the same walk).
