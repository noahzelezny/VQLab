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

# Qwen3.8-Flash-Next-VQ-5.5bpw

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

## Requirements

**This model needs an `mlx-lm` that has the `qwen4_exp` architecture, which no
released version has yet.** The architecture is in
[ml-explore/mlx-lm PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788),
still unmerged as of 2026-09-09. On a stock `pip install mlx-lm` you
will get:

```
ModuleNotFoundError: No module named 'mlx_lm.models.qwen4_exp'
```

That is the architecture missing, not a problem with this artifact. Until the
PR merges:

```bash
pip install git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1788/head

python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.8-Flash-Next-VQ-5.5bpw \
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
Budget ≈2.2 GiB extra resident when drafting is enabled. Through-exo throughput is
measured so far only on the 2.1bpw rung (23.7–25.5 tok/s, acceptance
0.82–0.88); the sidecar head is the same file on every rung.

## Changelog

### 2026-09-09 — runtime refresh

**Runtime refresh.** The bundled `model.py` is updated so downloaders run
exactly the code that was benchmarked; dense bundles now carry both runtimes.

- Faster prefill on affected geometries, measured per rung: a
  device-codebook kernel arm for large-codebook geometries (up to 1.46x on
  affected rungs), a ragged-subvector relaxation (up to 1.34x on affected
  rungs), a fused d8 arm, and a routing memo (≈2%). No blanket speedup is
  claimed across the lineup — gains apply only where the geometry engages
  the new paths.
- Output quality is unchanged: the kernel changes are bit-identical or
  1-ULP-equivalent, and the routing memo is bit-identical (logits checksum
  verified).
- Speculative decoding (MTP): on repos that ship
  `mtp-head-q6.safetensors`, the sidecar works with the exo fork branch
  `mtp-stage1` (github.com/noahzelezny/exo) — launch each node with
  `exo --mtp` (or set `EXO_MTP=1`). Note: with MTP enabled, exo serves
  requests sequentially (the batch engine has no MTP path), so leave it
  off for concurrent / multi-agent workloads.

### 2026-09 — bundle refresh

**Bundle refresh.**

`model.py` is updated:

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

**Memory, measured externally** (process RSS sampled at 5 Hz from outside,
M4 128 GB, weights ≈112 GiB on disk):

- **61.3 GiB** peak for load + 2048-token prefill + 128-token decode —
  expert routing concentrates, and untouched expert pages are never
  loaded from the mmap'd weights.
- **≈112 GiB** (the weight size) is the only figure that covers every
  possible routing pattern; budget it for worst-case or long, varied
  traffic.

Transient prefill allocations are returned to the OS as they free, so
peak tracks what the routing touches rather than a multiple of it.
Add ≈2.2 GiB when the MTP sidecar is enabled.

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

## Run it

```bash
pip install git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1788/head

python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.8-Flash-Next-VQ-5.5bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 512
```

The VQ runtime needs no patches — it ships inside the checkpoint as
`model.py` and stock `mlx-lm` executes it. The PR above is required only for
the base architecture; see Requirements.

## Speculative decoding (MTP)

This repo includes `mtp-head-q6.safetensors` (2.1 GiB): the model's own
multi-token-prediction head, quantized. Stock loaders ignore it — it
costs nothing on disk-to-RAM unless you opt in by name.

**When enabled it adds ≈2.2 GiB resident** (head weights + its cache) on
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
  --model TheDrainFlorist/Qwen3.8-Flash-Next-VQ-5.5bpw \
  --sidecar mtp-head-q6.safetensors
```

OpenAI-compatible API on localhost; `vqlab mtp-generate` for one-shot
CLI use. Without `--sidecar`, nothing about the model changes.

## Methodology

Fitted **data-free** from the bf16 checkpoint — k-means / Lloyd over weight
subvectors, seed 1234, no Hessian, no activation statistics, no calibration
corpus. Recipes are in the VQLab repo.

Quantization damage is not uniform across layers. A one-pass probe
(teacher and student streamed together, per-layer local damage measured
with no compounding) shows the same hot set on every rung of this family:
layer 1 dominates, a late band (31–39) follows, and the map is identical
across geometries (rank correlation 0.905, identical top-10). Upgrading
only those layers buys 15–24% KL for 3–4% size on the lower rungs; the
probe, the mixing, and the verdicts are all reproducible with VQLab
(`vqlab layer-leverage`, scatter fits via `fit-moe --vq-layers`).

## Verification

Release gates passed on this artifact before upload: file, index and
tokenizer checks, a verbatim match between the bundled runtime and its
source, and a generation smoke through the shipping runtime on Apple
Silicon. The upload path runs the gate itself and refuses to publish
without it.

## Limitations

- **This model needs an unreleased `mlx-lm`.** The `qwen4_exp` architecture is
  in [PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788), still unmerged
  as of 2026-09-09. See Requirements.
- **Stock exo cannot serve it** — upstream pins a released `mlx-lm` that lacks
  the architecture. Use our fork's `mtp-stage1` branch.
- **Through-exo throughput is measured on the 2.1bpw rung only**
  (23.7–25.5 tok/s, acceptance 0.82–0.88). The sidecar head is the
  same file on every rung, but this rung's cluster numbers are not
  measured.
- **Rank the table by KL, not perplexity.** Perplexity is an aggregate over
  finite text and absorbs offsetting errors; several rungs read within noise of
  the teacher on perplexity while differing by an order of magnitude in KL.
- **The affine comparators are our own conversions**, not community builds.
- **Perplexity is not comparable across model families** — only within this
  table, which is one instrument on one corpus set.

## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

## Provenance

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — released under the **Qwen Community License 1.0**.
This is a quantized derivative and inherits that licence; using it
means accepting the base model's terms.
Quantization: TheDrainFlorist, 2026.

Local artifact: `qwen4exp_vq_packed_d2k1024`.

Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab).
