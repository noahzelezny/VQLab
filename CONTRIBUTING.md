# Contributing

Read [METHODOLOGY.md](METHODOLOGY.md) first — it is the contract. The short
version of what this project asks of contributed results and code:

## Results

- **Pre-register**: state the prediction and the reading grid before the
  number exists. Falsified predictions are recorded as falsified.
- **Gate before believing**: outlier gate on the assembled artifact
  (ideally on a different machine), packed bytes only, smoke-generation
  through the shipping kernel before "releasable".
- **Floors before margins**: measure the seed-noise floor at your geometry
  (n≥3 fits) before claiming any small difference.
- **Name your instruments**: every comparison row names the artifact, the
  instrument, and how the comparator was produced. Record the resolved
  interpreter stack (python, mlx version, the `.so` that actually loaded)
  per fit.
- **Stamp provenance**: `vqlab manifest write` after gating; `check`
  before citing.

## Code

- New fitters start from the guard list in
  [docs/ONBOARDING.md](docs/ONBOARDING.md), not from a working fit.
- Any `mx.load` → save path needs the lazy-read cure: `with
  mx.stream(mx.cpu):` + `mx.eval` inside the block + all-zero assertion.
- Every new gate must FAIL on a known-bad input and PASS on a known-good
  before its verdict is trusted.
- Run `vqlab selftest` before sending a change and after pulling one; it
  exercises every gate in both directions in under a minute.
- Run `scripts/check_scripts_sync.sh` before long chains — stale scripts on
  a second box are silent divergence.

## New model families

Run the two-hour characterisation pass in
[docs/ONBOARDING.md](docs/ONBOARDING.md) before fitting anything: establish
a deterministic instrument, profile the weight geometry, run the init
sweep. A family whose scores are non-deterministic cannot rank quants —
that is why gemma-family code ships here with no quality claims attached.
