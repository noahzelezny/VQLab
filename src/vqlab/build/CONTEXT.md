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
- `stream-convert` refuses to run when knurlogic is not importable (stock mlx-lm would serve the architecture: different sanitize dtypes, e.g. qwen3_5 norms bf16 vs float32). Use the vqlab venv, or set `VQLAB_VENDORED_ARCH=0` to build on stock mlx-lm deliberately. Every model load (`family/model_loader.py`) installs Knurlogic's arch finder itself, however the tool was invoked.
- Preflight every new writer on a `vqlab slice` of the teacher first; a 4-layer slice caught two stream-convert bugs in minutes (OPERATOR-NOTES-2026-10-03 sec 4).
- The MoE code packing rule is `core/artifacts/codepack.py` (fit-moe --pack shares it); `vqlab.assemble.codepack` is an alias for one release.
- This folder was `assemble/` until 2026-10-10 (renamed to match `vqlab build ...`). `vqlab.assemble` and every `vqlab.assemble.<mod>` (incl. `build_dense_vq`, now `build_dense`) are deprecated aliases for one release: same module objects, with a DeprecationWarning. The generic `plan()` in mix / graft_extras / reskeleton is now `plan_mix` / `plan_graft` / `plan_reskeleton` (`plan` kept as an alias).

## Files
- `build_dense.py`: `build dense`: a runnable dense VQ artifact from a quantized base + VQ fits
- `graft_extras.py`: `build graft-extras`: copy tensors a text-only VQ artifact lacks from the official release
- `graft_vision.py`: `build graft`: graft the vision tower into a VQ artifact
- `loo_bands.py`: `plan loo-bands`: leave-one-out band hybrids of a VQ build, plus the KL queue
- `minibase.py`: `build minibase`: a fit base holding only one layer band's shards (symlinks)
- `mix.py`: `build mix`: an artifact from per-layer-band sources, shard by shard
- `mtp_extract.py`: `build mtp-extract`: a model's MTP head from its source checkpoint into one graft file
- `mtp_graft.py`: `build mtp-graft`: an MTP head in the layout a native qwen4_exp runtime expects
- `mtp_head_ds4.py`: `build mtp-head-ds4`: DeepSeek-V4-Flash's draft head for a VQ trunk
- `mtp_pack.py`: `build mtp-pack`: a bf16 MTP graft -> a quantized drafting sidecar
- `pack.py`: `build pack`: pack an unpacked VQ codes artifact to sub-byte codes
- `pack_dense.py`: `build pack-dense`: pack a dense VQ artifact's codes (new output dir)
- `pack_ple.py`: `build pack-ple`: pack PLE codes to true 11-bit width, row-aligned
- `ple_swap.py`: `build ple-swap`: swap an artifact's PLE tables for another rung's, byte for byte
- `reskeleton.py`: `build reskeleton`: put a VQ build's expert tensors on another build's skeleton
- `sanitize_stream.py`: `build sanitize-stream`: convert to MLX layout one layer at a time through Model.sanitize
- `slice_teacher.py`: `build slice`: a small real slice of any teacher, for preflighting a writer
- `splice_ple.py`: `build splice-ple`: splice VQ PLE codes into the packed artifact
- `stream_convert.py`: `build stream-convert`: streaming affine convert for models bigger than RAM
- `teacher_prep.py`: `build teacher-prep`: an official release -> an exact MLX-layout teacher
- `unpack_dense.py`: `build unpack-dense`: rewrite a packed dense VQ artifact with plain codes (diagnostic)
