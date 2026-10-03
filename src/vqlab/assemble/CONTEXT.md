# assemble/ — Assemble: codes -> an artifact that loads

## Inputs
Fit outputs, a quantized base / struct skeleton.

## Process
`vqlab pack` (or `fit-moe --pack`, which packs as each shard is written, byte for byte the same), `pack-dense`, `pack-ple`, `unpack-dense` (diagnostic twin), `splice-ple`, `stream-convert`, `build-dense`, `graft`, `ple-swap`, `slice` (a few real teacher layers + embeddings/norm/head, config adjusted, for preflighting a writer).

## Outputs
A new artifact directory with shards, `config.json` (`vq_modules` + `pack_bits`) and an index.

## Rules that bite
- Outputs go under the configured storage (`vqlab.config`: scratch, models, fit store), never the repo or an arbitrary path.
- Same recipe != same bytes: an mlx version change moved 35B skeleton scales up to 12.8%. Record `env.mlx`.
- Ragged packing (nsub not a multiple of 32) wastes bytes; `geo-build` refuses it.
- Preflight every new writer on a `vqlab slice` of the teacher first; a 4-layer slice caught two stream-convert bugs in minutes (OPERATOR-NOTES-2026-10-03 sec 4).
