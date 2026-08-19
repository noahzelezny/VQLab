---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- qwen3.6
---

# Qwen3.6-35B-A3B-VQ-tail30-d2K512

**17.9 GiB at bf16-parity perplexity — half the size of the 8-bit, on a 32 GB Mac.**

A vector-quantized build of
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) for Apple
Silicon. Stock `mlx-lm`, no patches.

## Measured results

Referee corpus, 2048-token windows, scored against the bf16 teacher on the
same files with an unmodified `mlx-lm`:

| build | size | perplexity | vs bf16 | top-1 agreement |
|---|---|---|---|---|
| bf16 | 70 GiB | 4.7215 | 1.000x | 100% |
| mlx-community 8-bit | 35 GiB | — | 0.999x | 96.18% |
| **this model** | **17.9 GiB** | **4.6812** | **0.991x** | 90.75% |
| mlx-community 4-bit | 19 GiB | — | 1.041x | 85.61% |

**At parity with bf16 on perplexity, at 26% of its size** — and smaller than
the 4-bit while being ~5% better on ppl.

**Read 0.991x as "at parity", not "better than bf16".** Mild quantization
slightly reducing referee perplexity is a known effect on this corpus (the
8-bit reads 0.999x). The corpus is finite; treat anything in 0.99-1.00x as
indistinguishable from the teacher.

Perplexity is corpus-specific — compare only against models scored on the
same files, never across harnesses.

Note the two metrics disagree in emphasis: ppl says parity, top-1 agreement
says 90.75% vs the 8-bit's 96.18%. For a 256-expert MoE that is expected —
discrete routing flips swap one plausible token for another, which
perplexity absorbs and argmax-agreement punishes. Both numbers are honest;
quote both.

## Runtime

Single M3 Ultra, macOS, stock `mlx-lm`, GPU otherwise idle:

| | |
|---|---|
| load time | ~9 s |
| peak memory | 18.0 GiB |
| decode | **~46 tok/s** |

Comfortable on a 32 GB machine with context headroom — the 8-bit needs 48 GB+.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.6-35B-A3B-VQ-tail30-d2K512 \
  --prompt "Write a Python function that ..." \
  --max-tokens 512
```

## How it was built

A **depth schedule**, not a uniform quantization. Layers 10-39 (the "tail")
carry vector-quantized experts at d=2/K=512; layers 0-9 stay at d=4/K=2048.
Non-expert tensors are 8-bit throughout.

The tail depth and the tail codebook were tuned separately, and that turned
out to matter: a *richer* d2-K2048 tail produced a **larger and worse**
artifact (20.7 GiB, 1.000x) than this one. The tail had been sized while the
geometry was d4 and nobody re-tuned it afterwards. Going cheaper again
(d2-K256) overshot — 16.5 GiB but 1.002x — so K512 is the knee, found by
bracketing rather than by extrapolation.

## Verification

Every tensor was decoded **from the published artifact** and compared
against the bf16 source; no tensor exceeds 3x the artifact's own median
reconstruction error. Packing was separately verified as a pure
representation change: the packed model scores identically to the unpacked
one to three decimals.


### Multi-machine (exo) note

These artifacts fit on one machine, but if you shard them across an
[exo](https://github.com/exo-explore/exo) cluster anyway, one guard is
required: VQ codebooks must **replicate rather than slice**. Without it,
exo's tensor parallelism splits the codebook silently and the model
generates fluent garbage that reads as "a broken quant." The guard is
bundled in this artifact's `model.py`; upstream fix submitted as
[exo PR #2268](https://github.com/exo-explore/exo/pull/2268).

## Limitations

- Top-1 agreement is 90.75% against the 8-bit's 96.18%. If your workload is
  sensitive to exact token choice rather than distributional quality, the
  8-bit is measurably closer to bf16.
- Perplexity was measured on one corpus. A code-heavy or non-English
  workload may rank these builds differently.
- No blind human-preference evaluation was run on this model (unlike the
  gemma builds in this collection); the claim here rests on perplexity,
  which is valid for this family.
