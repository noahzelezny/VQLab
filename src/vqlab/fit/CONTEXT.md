# fit/ — Fit: teacher weights -> codebooks + codes

## Inputs
A bf16 teacher (on the HDD, never re-downloaded), a family, a geometry (d, K) or a geomap.

## Process
`vqlab fit-moe`, `fit-dense`, `fit-ple`, `geo-build` (refit only the named modules; every other module keeps its shipped bytes), `harvest-parts`, `fits` (the fit store: index / list / file).

## Outputs
A new directory (never in place) carrying `vqlab_provenance.json`: the fitter settings, seed, inputs and per-module origin. `geo-build` also keeps `origins.json` in its parts dir.

## Rules that bite
- **Look in the fit store first**: `vqlab fits list --family F --teacher T --layers A-B --geom dD-KK`. `geo-build --pool` reuses matches automatically. Every fitter files what it fits (`core/fitstore.put`).
- Seeded by default (1234). `--seed -1` is the only way to get a random draw, and the build record says so.
- `--relerr-abort` scales with K (K2048 ~0.19, K256 ~0.31, K128 ~0.46). Set it per geometry.
- Fitted parts go to `<fits>/`. Never refit what was already paid for.
- Load source tensors on the CPU stream with `mx.eval` INSIDE the block (FINDINGS IV.1).
