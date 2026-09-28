# experimental/skipzero/ — vq-skipzero: drop dead VQ rows on disk (EXPERIMENT)

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
