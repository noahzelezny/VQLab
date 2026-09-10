---
language:
- en
license: gemma
library_name: mlx
pipeline_tag: image-text-to-text
base_model: google/gemma-4-e4b-it
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- gemma-4
- multimodal
---

![chart](chart_e4b_vqple.png)

# gemma-4-e4b-it-VQ-PLE

**7.39 GiB — the community 8-bit, with a better embedding table.** This is
mlx-community's `gemma-4-e4b-it-8bit` with exactly one change: the 5.25 GiB
per-layer-embedding table (35% of the model's bytes) is replaced by a
vector-quantized version at 5.75 bits/weight. Everything else — attention,
MLPs, norms, towers — is byte-identical to the artifact you already know.

That one swap makes the model **measurably closer to bf16 than the 8-bit
it came from**, at 1 GiB less disk, ≈1 GB less peak RAM, and decode speed
at parity.

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

### 2026-09 — bundle repair

**Bundle repair — this model was broken for downloaders until today.**

Two stacked defects, both ours, both fixed:

1. The shipped `model.py` was incomplete (it imported runtime modules
   that only existed in our development environment), so on a stock
   install the model could not load at all. Repairing that exposed:
2. The VQ embedding decode returned fp16 into a bf16 model; MLX's
   promotion rules turned the whole forward fp32 from layer 1 on, which
   at this model's attention geometry crashes the Metal kernel on any
   real prompt. One cast fixes it.

**No published score was ever affected** — scoring ran through the
correct decode path; only the bundled runtime was broken. The bundle is
now self-contained, gated by a load-and-generate check in a
verified-clean environment, and generates coherently at length on both
M3- and M4-class machines.

**Peak memory, measured externally:** process RSS sampled at 5 Hz across
load + 2048-token prefill + 128-token decode peaks at **7.5 GiB** — the
on-disk size. Peak ≈ resident.

## Measured results

All numbers on the same instruments, same corpus, same teacher cache;
"incumbent" = mlx-community gemma-4-e4b-it-8bit as shipped.

| | **this artifact** | incumbent 8-bit |
|---|---|---|
| size on disk | **7.39 GiB** | 8.38 GiB |
| KL to bf16 (literary corpus) | **7.451 mnats/token** | 8.149 mnats/token |
| top-1 agreement with bf16 | 95.70% | 95.70% |
| litbench (cyclic, generative, n=104) | 81.73% | 84.62% |
| — paired McNemar | 7 discordant items, 5–2, p=0.45 — statistically indistinguishable | |
| decode | 89.3 tok/s | 89.3 tok/s |
| prompt processing (2k prompt) | **3570 tok/s** | 3499 tok/s |
| peak memory (short chat) | **7.1 GB** | 8.0 GB |

Runtime rows measured 2026-09 on an M3 Ultra (mlx 0.32.2, 256-token
greedy decode, repeated runs within 0.6%). The 2026-09 bundle's kernel
refresh closed what used to be an ≈8% decode deficit against the 8-bit.

Honest summary: closer to the bf16 teacher on the precise instrument (KL),
indistinguishable on the noisy one (litbench, n=104 cannot resolve a
3-point gap — SE ±3.7), decode at parity, ≈12% less RAM, 1 GiB less
disk. There is no axis left on which the 8-bit is the better download.

## Run it

```bash
pip install 'mlx-lm>=0.31.3'
mlx_lm.chat --model <this artifact>
```

The artifact is self-contained (`model.py` ships inside it); stock mlx_lm
loads it with no extra code. It also loads STRICTLY — the upstream 8-bit
artifact ships 126 tensors for KV-shared layers that mlx_lm never
instantiates and silently drops; those are removed here.

## Why the embedding table

We first vector-quantized everything (MLPs + embeddings) and LOST to the
8-bit decisively (20.8 vs 8.1 mnats): e4b's MLP weights do not tolerate
5.75-bit VQ even though their reconstruction error looks excellent — fit
error and output damage are different quantities. An ablation split the
damage: the MLP contributed essentially all of it, and the VQ embedding
table alone was *better* than its 8-bit affine counterpart. So this
artifact keeps the 8-bit MLPs and ships only the swap that wins.

Embeddings are the friendly case for VQ at runtime too: a lookup decodes
only the rows a batch touches — no matmul kernel, no full-table
materialization, which is where the RAM saving comes from.

## No calibration data

The VQ fit is pure weight-space k-means against the bf16 tensors: no
calibration corpus, no forward passes, no distillation. 154 seconds of
codebook fitting on an M3 Ultra. Every number above was measured after,
not optimized for.

## Methodology

Fitted **data-free** from the bf16 checkpoint — k-means / Lloyd over weight
subvectors, seed 1234, no Hessian, no activation statistics, no calibration
corpus. Recipes are in the VQLab repo.

Only the per-layer-embedding table is replaced; every other tensor is
carried over from the 8-bit conversion unchanged. See *Why the embedding
table* above for why that is the high-leverage target.

## Verification

- VQ table verified decode-side against the bf16 source (relerr 0.0296,
  uniform across all 262,144 rows).
- Packed codes verified bit-exact against the unpacked reference, and the
  packed artifact reproduces the KL score to the third decimal.
- Corrupting the codebook garbles generation — the VQ path is provably
  live, not a fallback.

## Limitations

- ≈8% slower decode and ≈20% slower prefill than the 8-bit incumbent.
- litbench point estimate is 3 points below the incumbent; the paired test
  says noise (p=0.45), and the KL says closer-to-teacher, but if your use
  case resembles literary MC comprehension specifically, measure your own.
- Vision/audio towers are carried unchanged from the incumbent artifact;
  vision was not re-benched here.

### Multi-machine (exo) note

If you shard this across an exo cluster: VQ codebooks must **replicate,
not slice**. The guard is bundled in `model.py`; upstream fix is
[exo PR #2268](https://github.com/exo-explore/exo/pull/2268).

## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)
## Provenance

Base model: [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it) — governed by the **Gemma Terms of Use**.
This is a quantized derivative and inherits that licence; using it
means accepting the base model's terms.
Quantization: TheDrainFlorist, 2026.

This build starts from
[mlx-community/gemma-4-e4b-it-8bit](https://huggingface.co/mlx-community/gemma-4-e4b-it-8bit)
and replaces its per-layer-embedding table; that conversion's
authors are credited accordingly.

Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab).
