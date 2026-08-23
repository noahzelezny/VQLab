# MoEMash

**Custom size-targeted vector-quantized (VQ) builds of large models on Apple
Silicon, with MLX.** Name a byte budget, price the recipe before fitting,
fit it data-free, pack to true bit-width, verify, and serve through stock
`mlx-lm` — the runtime ships inside the artifact.

The method and every number below are documented in the companion paper,
[*Data-Free Vector Quantization Beats Affine Quantization at Matched
Bytes*](paper/PAPER.md). Released artifacts are published under
[TheDrainFlorist](https://huggingface.co/TheDrainFlorist) on Hugging Face
with their VQ runtimes bundled in-checkpoint (stock `mlx-lm`, no patches). All sizes are measured packed bytes; every margin is stated against
a measured seed-noise floor; see [METHODOLOGY.md](METHODOLOGY.md) for the
rules that keep these numbers honest.

## Measured results

**397B MoE (Qwen3.5-397B-A17B), streaming-referee perplexity, prose/code:**

| build | GiB | prose ppl | code ppl |
|---|---|---|---|
| d8/K16384 | 100.97 | 3.0591 | 2.6728 |
| flat K256 (2.4bpw) | 111.62 | 2.7655 | 2.6383 |
| **flat K512** | **122.31** | **2.5634** | **2.6123** |
| flat K2048 (flagship) | 143.68 | 2.3410 | 2.5963 |
| calibrated 2.6-bit comparator | 120.6 | 3.1843 | 2.6667 |
| calibrated 3.5-bit comparator | 165.6 | 2.3614 | 2.6005 |

- **flat K512 beats the calibrated 2.6-bit build by 0.62 prose ppl (46x the
  noise floor) and on code (3.4x floor), at +1.7 GiB** — the cleanest
  like-for-like on the ladder.
- **d8/K16384 beats the same comparator while being 19.6 GiB smaller**
  (prose margin 9.3x floor). It is the ~101 GiB build a 128 GB Mac can hold.
- **The flagship matches the calibrated 3.5-bit build at 21.9 GiB smaller**
  (quality indistinguishable at the floor — we claim the bytes, not a
  quality edge).

**35B MoE (Qwen3.6-35B-A3B), KL-to-bf16:** d4/K8192 measures **53.0 mnats /
89.55% top-1 at 14.84 GiB packed** vs the community 4-bit's 78.6 / 85.61% at
19.0 GiB — a 32% KL reduction at 4.16 GiB smaller.

**Dense 27B (KL-to-bf16 + ppl):** the recipe is not an MoE phenomenon.
d2/K512 beats the 4-bit affine conversion by 27.8% KL (6.1x floor) and
+1.28 pp top-1 at 4-bit-class size; d4/K1024 beats q3 on both metrics at
0.35 GiB less.

**Where the method ends, measured:** the VQ advantage lives at the low-bpw
end — wins measured at 2.0–4.5 bpw across three families; the VQ/affine
crossover is bracketed at 4.5–6.0 bpw on the dense 27B; by 8 bits affine is
essentially lossless and the advantage is gone. Prefill remains ~0.5x affine
at 35B. We publish these fences as measured boundaries, not caveats.

## The differentiating feature: size targeting

Flat rungs leave gaps. MoEMash prices an artifact **before fitting it**:

```bash
moemash price --family qwen397b --budget-gib 108
```

Two measured size models back this: the 397B harvest form
(`new = base − 1.87 GiB × shallow_bits_harvested`, 6-for-7 within ±0.4 GiB)
and the dense composition form (`total = codes + scales + carry`, closed to
≤0.003 GiB across three builds and two geometries). Harvesting shallow-layer
K back is ~2x the byte-efficiency of stepping down a flat rung — it buys the
sizes between rungs. It is never free and never beats a flat rung at the flat
rung's own size; the pricer tells you both.

## Pipeline

```
fit → verify (outlier gate) → pack → graft (vision) → release checks
    → smoke-generate through the shipped kernel → score
```

```bash
pip install -e .

moemash fit-moe   --family qwen397b --model <bf16-or-base> --out fits/K256 ...
moemash fit-dense --model <src> --out fits/d2K512 ...
moemash verify    <artifact> --outlier 3.0        # BEFORE believing any score
moemash pack      <artifact> --out <packed>       # true bit-width, measured size
moemash graft     <packed> --vision <bf16>        # MoE vision tower, bf16
moemash check     <packed>                        # release + bundle gates
moemash smoke     <packed>                        # one token, fused path, resident
moemash score     <packed> --instrument referee   # or kl
moemash manifest  write <packed>                  # stamp provenance
```

Every subcommand is a thin wrapper over a standalone script in
`src/moemash/`; `moemash <cmd> --help` shows the full surface. See
[REPRODUCING.md](REPRODUCING.md) for the exact commands behind each paper
table row.

## Requirements

- Apple Silicon Mac; RAM sized to the artifact for fit/smoke (scoring
  streams and can exceed RAM). The 397B fits used 96–128 GB machines.
- Python ≥ 3.12, `mlx`, `mlx-lm` (stock — artifacts bundle their own
  runtime), `numpy`, `safetensors`.

## Honesty rules baked into the tool

- The outlier gate refuses to be skipped quietly: scoring an ungated
  artifact is on you, and the docs say what it cost us.
- Sizes are computed from packed shards on disk, never from index metadata.
- The fitter aborts on per-geometry relative-error thresholds and never
  writes in place (fits resume).
- `bundle-accept` tests the runtime copy *lifted from the artifact*, not
  whatever your import path resolves to.

## Known scope limits

- MLX/Metal only; kernel conclusions (threadgroup ceiling, d8 decode tax)
  are Apple Silicon specific.
- Families onboarded: Qwen3.5-397B-A17B, Qwen3.6-35B-A3B (MoE),
  dense Qwen 27B. Gemma-family fitting code is included, but no quality
  claims are made for it: its raw perplexity is invalid as a property of
  the model, which makes scoring non-deterministic.
- New family? Read `docs/ONBOARDING.md` (the two-hour characterisation pass
  to run before fitting anything).
