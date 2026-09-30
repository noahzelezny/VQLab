# fit/ — Fit: teacher weights -> codebooks + codes

## Inputs
A bf16 teacher (on the HDD, never re-downloaded), a family, a geometry (d, K) or a geomap.

## Process
`vqlab fit-moe`, `fit-dense`, `fit-ple`, `geo-build` (refit only the named modules; every other module keeps its shipped bytes), `harvest-parts`, `fits` (the fit store: index / list / file).

## Outputs
A new directory (never in place) carrying `vqlab_provenance.json`: the fitter settings, seed, inputs and per-module origin. `geo-build` also keeps `origins.json` in its parts dir.

## Rules that bite
- `vqlab fit-additive` fits C1+C2 (default 128x128) and writes them EXPANDED as ordinary d4-K16384 parts + origins.json; assemble with `geo-build --reuse`. Recipe init says "additive", so `--pool` never mixes it into ordinary builds. Compare against a true d4-K16384 refit at identical bytes with paired KL; weight relerr against K256 is not a fair comparison (different code width).
- `vqlab reselect calibrate|apply`: G-aware code re-selection (F67-F75) on a fixed codebook, block-diagonal Gram from self-generated text. F78 rejected it under the old ppl gate; it is re-tested under the F118 KL gate, ppl printed with its sign. Never filed in the fit store (its codes are not a k-means fit).
- **How a fit uses its codebook**: `vqlab fits census` (exact per-entry counts from the codes, cached in `<store>/census.jsonl`): dead entries, entropy, zero-scale groups (the teacher's near-zero rows).
- **Look in the fit store first**: `vqlab fits list --family F --teacher T --layers A-B --geom dD-KK`. `geo-build --pool` reuses matches automatically. Every fitter files what it fits (`core/fitstore.put`).
- Seeded by default (1234). `--seed -1` is the only way to get a random draw, and the build record says so.
- `--relerr-abort` scales with K (K2048 ~0.19, K256 ~0.31, K128 ~0.46). Set it per geometry.
- Fitted parts go to the fit store (`vqlab.config.fit_store`). Never refit what was already paid for.
- Load source tensors on the CPU stream with `mx.eval` INSIDE the block (FINDINGS IV.1).
