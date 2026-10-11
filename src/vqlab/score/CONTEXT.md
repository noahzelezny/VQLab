# score/ — Score: how much damage does it carry?

## Inputs
An artifact, per-corpus teacher caches, and the house corpora in `referee/`.

## Process
`vqlab kl-ladder` (THE release gate: paired, three corpora at 12288, |t|>2), `kl-pair` (pairs two already-scored runs, zero GPU), `kl`, `score` (ppl: printed, not gated), `tasks` (lm-eval task benchmarks, layer-streamed), `kernel-truth` (which rounding is closer to an exact float64 reference), `act-stats` (how often each clamp in a family's parity checklist fires on the house corpora; GPU, the family plugin's hook), `runtime-equiv` (one slice through two interpreters: bitwise equal or max |logit diff|; run whenever either env moves).

## Outputs
Per-position arrays and JSON records. kl-ladder resumes from them.

## Rules that bite
- One harness: never compare numbers across scoring paths or batchings.
- One numerics build: every cache, record and per-position sidecar carries `numerics` (mlx, mlx-lm, arch sha, scorer variant; `family/version_stamp.py`). Scoring REFUSES a cache from another build (`--allow-build-mismatch` records the override); `kl-pair` refuses arrays whose sidecars do not prove the same cache and build (`--allow-unverified-pair`).
- Perplexity is deterministic. A noise floor needs a second INDEPENDENT fit.
- The KL gate does not exercise the decode kernels (F137, F164). Say which path a number describes.
- Corpora moved here from `src/vqlab/referee/`. Paths recorded before the move resolve through `_layout.legacy_path`.
- `check_pin` lives in `core/artifacts/pin_record.py` (still importable from `gate.pin`); ClampCounter/corpus_ids live in `core/source/clamps.py` (re-exported from `act_stats`). score imports neither gate nor bench.

## Files
- `act_stats.py`: `score act-stats`: how often each clamp in a family's parity list fires
- `kernel_truth.py`: `score kernel-truth`: which dense VQ path is closer to exact float64
- `kernel_truth_moe.py`: `score kernel-truth-moe`: the MoE twin of kernel-truth
- `kl_damage.py`: `score kl`: quantization damage as KL from the model's own bf16 output
- `kl_ladder.py`: `score kl-ladder`: rank rungs by KL against per-corpus teacher caches (the release gate)
- `kl_pair.py`: `score kl-pair`: paired KL between two arms scored in separate runs
- `ppl_resident.py`: resident referee ppl through the runtime the artifact ships (`python -m vqlab.score.ppl_resident`; alloc-sweep scores with it; `scripts/score_ppl_resident.py` is a wrapper)
- `referee/score_streaming.py`: `score ppl`: the single-box streaming ppl referee over the house corpora
- `runtime_equiv.py`: `score runtime-equiv`: one model slice through two interpreters, bitwise or not
- `score_tasks_streaming.py`: `score tasks`: layer-streamed task benchmarks for models larger than RAM
- `stream_score.py`: `score stream-score`: the layer-streamed scorer kl-ladder runs; builds teacher caches
