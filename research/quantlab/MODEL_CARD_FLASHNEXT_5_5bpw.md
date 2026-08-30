---
language:
- en
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- qwen3.8
---

# TheDrainFlorist/Qwen3.8-Flash-Next-VQ-5.5bpw

**111.6 GiB — a 335 GiB frontier MoE on 128 GB machines (tight).**

A data-free vector-quantized build of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(180B total / 10-of-512 active, 51.2B n-gram PLE, vision) for Apple
Silicon. Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint as `model.py`. Built with [VQLab](https://github.com/noahzelezny/VQLab).

MoE experts at d=2/K=1024 (10-bit packed rows), PLE at d=8/K=4096. Flat allocation — at this size the leverage mix measured as a wash and is not shipped.

The affine builds compared against below are our own conversions made with
the same tooling, scored on the same instrument.

![where these releases sit](chart_ladder.png)

## Measured results

Prose referee, 2048 tokens; KL against the bf16 teacher's cached top-64
(captured mass 0.963 for every row — same cache, same positions). All
sizes include the 333-tensor bf16 vision tower (0.84 GiB).

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | perplexity |
|---|---|---|---|---|
| affine q3 (ours) | 75 GiB | 1083.4 | 61.9% | 12.850 |
| affine q4 (ours) | 96 GiB | 293.9 | 79.6% | 6.453 |
| affine q5 (ours) | 116 GiB | 91.7 | 87.5% | 5.243 |
| **this model** | **111.6 GiB** | **34.1** | **94.1%** | **5.245** |
| affine q6 (ours) | 137 GiB | 52.8 | 91.6% | 4.916 |
| affine q8 (ours) | 178 GiB | 27.1 | 94.9% | 5.197 |
| bf16 teacher | 335 GiB | 0 | 100% | 5.166 |

Additional corpora (perplexity): code 1.898 (public mlx corpus,
pinned manifest), literary 7.636 (Gutenberg). Teacher reads 1.902 / 7.664.

**Rank these by KL, not perplexity.** Perplexity is an aggregate over
finite text and absorbs offsetting errors; KL measures distance to the
teacher's distribution directly. Several rungs here read within noise of
the teacher on perplexity while differing by an order of magnitude in KL.

## The leverage mix

Quantization damage is not uniform across layers. A one-pass probe
(teacher and student streamed together, per-layer local damage measured
with no compounding) shows the same hot set on every rung of this family:
layer 1 dominates, a late band (31–39) follows, and the map is identical
across geometries (rank correlation 0.905, identical top-10). Upgrading
only those layers buys 15–24% KL for 3–4% size on the lower rungs; the
probe, the mixing, and the verdicts are all reproducible with VQLab
(`vqlab layer-leverage`, scatter fits via `fit-moe --vq-layers`).

## Provenance and gates

Fitted data-free from the bf16 checkpoint (k-means / Lloyd on weights,
seed 1234, recipes in the VQLab repo). Release gates passed on this
artifact: file/index/tokenizer checks, bundle-runtime verbatim match, and
a generation smoke through the shipping runtime on Apple Silicon.
exo-ready: config carries vision_config + image_token_id; the vision
tower is grafted bf16.

Local artifact: `qwen4exp_vq_packed_d2k1024`.
