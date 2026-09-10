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

# TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw

**45.0 GiB — a 335 GiB frontier MoE on 64 GB machines.**

A data-free vector-quantized build of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(180B total / 10-of-512 active, 51.2B n-gram PLE, vision) for Apple
Silicon. Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint as `model.py`. Built with [VQLab](https://github.com/noahzelezny/VQLab).

MoE experts at d=8/K=16384 (14-bit codes, padded-tail packed), PLE n-gram tables at d=8/K=256 (8-bit rows), and layers 0–1 upgraded to d=2/K=256 experts (the leverage mix — see below).

The affine builds compared against below are our own conversions made with
the same tooling, scored on the same instrument.

![where these releases sit](chart_ladder.png)


## Requirements — read before downloading

**This model needs an `mlx-lm` that has the `qwen4_exp` architecture, which no
released version has yet.** The architecture is in
[ml-explore/mlx-lm PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788),
still unmerged at the time of writing. On a stock `pip install mlx-lm` you
will get:

```
ModuleNotFoundError: No module named 'mlx_lm.models.qwen4_exp'
```

That is the architecture missing, not a problem with this artifact. Until the
PR merges:

```bash
pip install git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1788/head

python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 512
```

The VQ runtime itself needs no patches — it ships inside the checkpoint as
`model.py` and stock `mlx-lm` executes it. The PR is required only for the
base architecture.

**Distributed serving via [exo](https://github.com/exo-explore/exo):**
upstream exo pins a released `mlx-lm` that lacks `qwen4_exp`, so stock exo
cannot serve this model. Our [exo fork, branch
`mtp-stage1`](https://github.com/noahzelezny/exo) serves it, and speculative
decoding there is one opt-in knob: set `EXO_MTP=1` in the worker's
environment and it loads `mtp-head-q6.safetensors` and drafts; leave it
unset (the default) and the sidecar is never read — no memory cost.
Budget ~2.2 GiB extra resident when drafting is enabled. On a 64 GB
machine the trunk alone is 45.8 GiB, so the head fits but leaves little
headroom for anything else. Measured on this rung through exo (M4, single
node): 23.7–25.5 tok/s decode at acceptance 0.82–0.88.

## Recent changes (2026-09)

**Bundle refresh.** `model.py` is updated:

- Faster MoE expert kernels (threadgroup-occupancy fix, device-direct
  activation reads, template-free dispatch, and a simd_sum reduction).
  Measured on this family: decode 17.4 -> 18.8 tok/s (+8%, A-B-A, M3
  Ultra). All kernels verified bit-identical to the previous bundle
  except the reduction, which is 1-ULP equivalent with a measured
  quality delta of exactly zero (identical perplexity to 16 digits,
  KL 0.0 between old and new logits). `VQ_D8_SS=0` restores the
  bit-identical legacy reduction.
- The runtime caps MLX's buffer-reuse cache by default
  (`VQLAB_CACHE_LIMIT_GB` overrides, `0` disables), so long-prompt
  prefill no longer balloons transient memory — peak stays near
  resident instead of a multiple of it.
- The bundle loads under both `mlx-lm` and `mlx_vlm`.
- **Nothing you have downloaded breaks**: weights unchanged, only
  `model.py` and `config.json` keys replaced; prior revisions stay
  pinnable by commit hash.

**Peak memory, measured externally:** process RSS sampled at 5 Hz from
outside across load + 2048-token prefill + 128-token decode peaks at
**45.8 GiB** — the weight size; the runtime returns transient prefill
allocations as they free, so peak ≈ resident (without the MTP sidecar;
add ~2.2 GiB when enabled).

## Speculative decoding (MTP) — optional sidecar

This repo includes `mtp-head-q6.safetensors` (2.1 GiB): the model's own
multi-token-prediction head, quantized. Stock loaders ignore it — it
costs nothing on disk-to-RAM unless you opt in by name.

**When enabled it adds ~2.2 GiB resident** (head weights + its cache) on
top of the trunk, so budget for it. What you get, measured on this
artifact (M3 Ultra, greedy, 378-token runs): 18.1 -> 24.4 tok/s,
acceptance 0.77. The trunk verifies every drafted token by exact
rejection sampling, so the output distribution is exactly the base
model's. Prefill in the speculative path is chunked (2048-token chunks,
`VQLAB_PREFILL_CHUNK` overrides) with per-chunk eval, so long prompts
stay within the same memory envelope as plain decoding.

To use it, serve with VQLab:

```bash
git clone https://github.com/noahzelezny/VQLab && cd VQLab
python3 -m venv .venv && source .venv/bin/activate
pip install .
python -m vqlab.cli serve \
  --model TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw \
  --sidecar mtp-head-q6.safetensors
```

OpenAI-compatible API on localhost; `vqlab mtp-generate` for one-shot
CLI use. Without `--sidecar`, nothing about the model changes.

## Measured results

Prose referee, 2048 tokens; KL against the bf16 teacher's cached top-64
(captured mass 0.963 for every row — same cache, same positions). All
sizes include the 333-tensor bf16 vision tower (0.84 GiB).

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | perplexity |
|---|---|---|---|---|
| affine q3 (ours) | 75 GiB | 1083.4 | 61.9% | 12.850 |
| **this model** | **45.0 GiB** | **390.1** | **78.8%** | **5.903** |
| affine q4 (ours) | 96 GiB | 293.9 | 79.6% | 6.453 |
| affine q5 (ours) | 116 GiB | 91.7 | 87.5% | 5.243 |
| affine q6 (ours) | 137 GiB | 52.8 | 91.6% | 4.916 |
| affine q8 (ours) | 178 GiB | 27.1 | 94.9% | 5.197 |
| bf16 teacher | 335 GiB | 0 | 100% | 5.166 |

Additional corpora (perplexity): code 2.076 (public mlx corpus,
pinned manifest), literary 8.945 (Gutenberg). Teacher reads 1.902 / 7.664.

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

Local artifact: `qwen4exp_vq_packed_mixL01`.
