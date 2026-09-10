---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- qwen3.8
---

# TheDrainFlorist/Qwen3.8-27B-VQ-4.5bpw

**14.5 GiB — smaller than our 4-bit conversion, and closer to bf16.**

A vector-quantized build of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple Silicon.
Stock `mlx-lm`, no patches — the VQ runtime ships inside the checkpoint as
`model.py`.

At the time of release no MLX-format quantization of this model had been
published, so unlike our 397B and 35B releases, the affine builds compared
against below are our own conversions (methodology and configuration in
Comparators below).

![where these releases sit](qwen38_ladder.png)

## Recent changes (2026-09)

**Bundle repair — if `mlx-lm` gave you `ModuleNotFoundError`, this fixes it.**
The previous `model.py` depended on a module our internal environment
injected into `mlx-lm`, so on a stock install the model failed to load at
all. The bundle is now fully self-contained and gated by a
load-and-generate check in a verified-clean environment before upload.
The weights are unchanged; only `model.py` is replaced. The prior
revision remains pinnable by commit hash, but for this model there is no
reason to pin it.

The bundled runtime also picks up the 2026-09 kernel refresh: the dense
VQ kernels now read activations directly from device memory instead of
staging through threadgroup tiles (bit-identical outputs, verified), and
prefill evaluates incrementally. Decode is meaningfully faster than the
previous bundle and peak memory now equals the on-disk size (figures
below are re-measured on this revision).


## Measured results

Scored against the bf16 teacher on the same corpus with an unmodified
`mlx-lm`. All sizes include the 333-tensor bf16 vision tower (0.859 GiB),
carried by every build here.

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | perplexity |
|---|---|---|---|---|
| affine q2 (ours) | 8.69 GiB | 1426.9 | 46.1% | 16.435 |
| affine q3 (ours) | 11.82 GiB | 187.8 | 79.5% | 5.832 |
| **this model** | **14.45 GiB** | **40.3** | **90.1%** | 5.233 |
| affine q4 (ours) | 14.95 GiB | 45.8 | 89.8% | 5.206 |
| affine q6 (ours) | 21.21 GiB | 3.71 | 96.8% | 5.260 |
| affine q8 (ours) | 27.48 GiB | 1.25 | 98.5% | 5.241 |
| bf16 | 51.7 GiB | 0 | 100% | — |

This is the cleanest comparison on the ladder: against the 4-bit affine
conversion it is **0.50 GiB smaller and 12% closer to bf16** (40.3 millinats
against 45.8), with 0.3 points better token agreement. Smaller and better on
the same instrument, no trade to weigh.

**Rank these by KL, not perplexity.** On this instruction-tuned family
perplexity barely moves — every build from 3-bit upward sits between 5.19 and
5.35, inside the measurement's own noise — while divergence from the teacher
moves by a factor of forty across the same range. Perplexity is an aggregate
over finite text and absorbs offsetting errors; KL measures distance to the
teacher's distribution directly.

## Runtime

**Decode 23.7 tok/s · prefill ~103 tok/s · resident 14.2 GB.**
Apple M3 Ultra, `mlx-lm generate`, 2048-token prompt, 128 greedy tokens, no
drafting head. Resident is process RSS, which stays at about the 14.5 GiB
on disk — this runs in its own size.

On the same machine, prompt and harness, stock `Qwen3.8-27B-8bit` runs at
23.7 tok/s in 27.5 GB of weights.

Throughput on other Apple silicon differs a lot — that same stock baseline
measures 16.7 tok/s on an M4 Max — so compare within a machine, not across
these numbers.

> **Peak memory, measured externally (2026-09-03):** process RSS sampled
> at 5 Hz from outside the process across load + 2048-token prefill +
> 128-token decode peaks at **14.4 GiB** — equal to the on-disk size.
> The bundled runtime returns transient prefill allocations to the OS as
> they free (buffer-cache cap, `VQLAB_CACHE_LIMIT_GB` overrides) and the
> prefill graph is evaluated incrementally, so peak ≈ resident. Ignore
> any tool's cumulative "Peak memory" line; it counts a high-water mark
> of freed transients.

> **Update 2026-09-01 — re-download `model.py` if you pulled this before.**
> Earlier builds shipped a `model.py` that imported a module which only
> existed in our development environment. On a normal install this artifact
> either could not generate at all, or fell back to reconstructing every
> weight matrix on each token — 7.8 tok/s instead of 24.2.
> Two new fused kernels fix the fallback; prefill improved as well.
>
> **Weights are byte-for-byte unchanged**, the quantization is untouched, and
> no score in this card is affected — scoring uses the long-input path, which
> is unchanged. Decode changed toward higher numerical accuracy, so generated
> text will differ slightly from earlier runs.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.8-27B-VQ-4.5bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 512
```

## How it was built

Vector quantization of the dense MLP trio at **d=2, K=256**. Each 2-weight
subvector stores one 8-bit index into a per-tensor 256-entry fp16 codebook. With an fp16 scale
per (row, 64 weights) that comes to 4.25 bits per weight over the quantized
surface; everything else in the model is 8-bit.

Every quantized tensor uses this one geometry: no depth schedule, no mixed
allocation.

**Codebooks are fit in pure weight space** — k-means over the weight
subvectors, no Hessian, no activation statistics, no calibration corpus.

**The fit is not seeded.** k-means draws an unseeded subsample, so this
artifact is reproducible in recipe and geometry but not bit-for-bit. Margins
are therefore quoted against a measured fit-to-fit floor rather than against a
repeated build; on this family that floor is 2.085 millinats.

## Comparators

The affine rungs above are local conversions made with `mlx_lm.convert`
using default settings, verified uniform per-module.

## Where this stops paying

Above roughly 5 bits per weight the advantage reverses on this model: our
6-bit affine conversion reaches 3.7 millinats at 21.2 GiB, which no VQ rung we
measured approaches at that size. Builds larger than the ones released here
were measured and deliberately not published for that reason.

## Verification

Every tensor was decoded **from the published artifact** and compared against
the bf16 source; no tensor exceeds 3x the artifact's own median reconstruction
error. The bundled runtime is exercised as the executing copy in a venv
with stock `mlx-lm` and none of our modules installed — the check that
the pre-2026-09-01 `model.py` defect escaped, because it had been run
in a development environment where the missing module happened to
exist. Vision tower grafted from the base
checkpoint and verified key-for-key against the official index, including the
channels-last patch-embedding layout that a naive rename gets silently wrong.

## Limitations

- Perplexity cannot rank builds on this family — see above.
- No task-suite scores for this artifact.
- The affine comparators are our own conversions, not community builds.
- Above ~5 bpw affine wins outright on this model; this collection stops
  below that line deliberately.
