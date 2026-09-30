# assemble/ — Assemble: codes -> an artifact that loads

## Inputs
Fit outputs, a quantized base / struct skeleton.

## Process
`vqlab pack`, `pack-dense`, `pack-ple`, `unpack-dense` (diagnostic twin), `splice-ple`, `stream-convert`, `build-dense`, `graft`, `ple-swap`.

## Outputs
A new artifact directory with shards, `config.json` (`vq_modules` + `pack_bits`) and an index.

## Rules that bite
- Outputs go under the configured storage (`vqlab.config`: scratch, models, fit store), never the repo or an arbitrary path.
- Same recipe != same bytes: an mlx version change moved 35B skeleton scales up to 12.8%. Record `env.mlx`.
- Ragged packing (nsub not a multiple of 32) wastes bytes; `geo-build` refuses it.
