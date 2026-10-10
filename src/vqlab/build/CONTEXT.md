# build/ — Build: codes -> an artifact that loads

## Inputs
Fit outputs, a quantized base / struct skeleton.

## Process
`vqlab pack` (or `fit-moe --pack`, which packs as each shard is written, byte for byte the same), `pack-dense`, `pack-ple`, `unpack-dense` (diagnostic twin), `splice-ple`, `stream-convert`, `build-dense`, `graft`, `ple-swap`, `mtp-extract` / `mtp-head-ds4` / `mtp-pack` / `mtp-graft` (draft heads; `mtp_head_ds4.py` moved here from fit/ in untangle step 15, `vqlab.fit.mtp_head_ds4` aliases it for one release), `slice` (a few real teacher layers + embeddings/norm/head, config adjusted, for preflighting a writer).

## Outputs
A new artifact directory with shards, `config.json` (`vq_modules` + `pack_bits`) and an index.

## Rules that bite
- Outputs go under the configured storage (`vqlab.config`: scratch, models, fit store), never the repo or an arbitrary path.
- Same recipe != same bytes: an mlx version change moved 35B skeleton scales up to 12.8%. Record `env.mlx`.
- Ragged packing (nsub not a multiple of 32) wastes bytes; `geo-build` refuses it.
- Preflight every new writer on a `vqlab slice` of the teacher first; a 4-layer slice caught two stream-convert bugs in minutes (OPERATOR-NOTES-2026-10-03 sec 4).
- The MoE code packing rule is `core/codepack.py` (fit-moe --pack shares it); `vqlab.assemble.codepack` is an alias for one release.
- This folder was `assemble/` until 2026-10-10 (renamed to match `vqlab build ...`). `vqlab.assemble` and every `vqlab.assemble.<mod>` (incl. `build_dense_vq`, now `build_dense`) are deprecated aliases for one release: same module objects, with a DeprecationWarning. The generic `plan()` in mix / graft_extras / reskeleton is now `plan_mix` / `plan_graft` / `plan_reskeleton` (`plan` kept as an alias).
