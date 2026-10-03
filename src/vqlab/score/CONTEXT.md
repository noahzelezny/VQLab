# score/ — Score: how much damage does it carry?

## Inputs
An artifact, per-corpus teacher caches, and the house corpora in `referee/`.

## Process
`vqlab kl-ladder` (THE release gate: paired, three corpora at 12288, |t|>2), `kl-pair` (pairs two already-scored runs, zero GPU), `kl`, `score` (ppl: printed, not gated), `tasks` (lm-eval task benchmarks, layer-streamed), `kernel-truth` (which rounding is closer to an exact float64 reference), `runtime-equiv` (one slice through two interpreters: bitwise equal or max |logit diff|; run whenever either env moves).

## Outputs
Per-position arrays and JSON records. kl-ladder resumes from them.

## Rules that bite
- One harness: never compare numbers across scoring paths or batchings.
- One numerics build: every cache, record and per-position sidecar carries `numerics` (mlx, mlx-lm, arch sha, scorer variant; `core/numerics.py`). Scoring REFUSES a cache from another build (`--allow-build-mismatch` records the override); `kl-pair` refuses arrays whose sidecars do not prove the same cache and build (`--allow-unverified-pair`).
- Perplexity is deterministic. A noise floor needs a second INDEPENDENT fit.
- The KL gate does not exercise the decode kernels (F137, F164). Say which path a number describes.
- Corpora moved here from `src/vqlab/referee/`. Paths recorded before the move resolve through `_layout.legacy_path`.
