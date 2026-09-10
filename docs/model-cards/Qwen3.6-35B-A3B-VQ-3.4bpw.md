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
- multimodal
---

# Qwen3.6-35B-A3B-VQ-3.4bpw

**13.8 GiB (vision tower included) — beats the 19 GiB 4-bit on perplexity at 73% of its size.**

The compact build in this collection: a vector-quantized
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) for machines
where the 18.7 GiB parity build is too large. Stock `mlx-lm`.

![where these releases sit](qwen36_ladder.png)

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

The bundled `model.py` runtime is updated as part of
the 2026-09 refresh shared across the VQ lineup:

- Updated MoE expert kernels (uint32 code loads for the d2 geometry,
  template-free specialized dispatch). For the code geometries this
  model uses, outputs are verified equivalent to the previous bundle
  and the speed change is modest; the refresh is mainly for runtime
  consistency across the lineup.
- The bundle loads under both `mlx-lm` and `mlx_vlm`.
- **Nothing you have downloaded breaks**: weights are unchanged, only
  `model.py` is replaced, and the prior revision stays pinnable by
  commit hash.

## Measured results

| build | size | vs bf16 ppl | top-1 agreement |
|---|---|---|---|
| bf16 | 65.4 GiB | 1.000x | 100% |
| mlx-community 8-bit | 35 GiB | 0.999x | 96.18% |
| sibling parity build | 18.7 GiB | 0.991x | 90.75% |
| **this model** | **13.8 GiB** | **1.029x** | **87.33%** |
| mlx-community 4-bit | 19 GiB | 1.041x | 85.61% |

**+2.9% perplexity against the 4-bit's +4.1%, at 73% of its size** — and the
gap widens on a code corpus (+1.9% vs the 4-bit's +4.6%, same harness).

This is a genuine quality step down from the parity build — it costs ≈3.8%
perplexity to save 4.9 GiB. If you have the RAM, take the 18.7 GiB one.

Perplexity is corpus-specific; compare only against models scored on the
same files.

## Runtime

Single M3 Ultra, macOS, stock `mlx-lm`, GPU otherwise idle:

| | |
|---|---|
| peak memory | 13.1 GiB |
| decode | **≈66 tok/s** |

*Measured on an M4 Max (128 GB), mlx-lm, 120-token greedy generation.*

Runs on a 16 GB machine at short context; comfortable on 24 GB+.

## Run it

```bash
pip install 'mlx-lm>=0.31.3'
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.4bpw \
  --prompt "Explain the difference between a mutex and a semaphore." \
  --max-tokens 512
```

## Methodology

Uniform vector quantization of the experts at **d=4, K=2048** (3.0
bits/weight packed); non-expert tensors at 8-bit. Experts are ≈92% of the
model, so that is where the bytes are.

Raising K further does not pay: K=4096 bought +0.55 agreement points for
+0.9 GiB, which is why this rung stops here.

## Verification

Every tensor decoded from the published artifact and compared against the
bf16 source; nothing exceeds 3x the artifact's own median reconstruction
error. Packing verified as a pure representation change — packed and
unpacked score identically to three decimals.

### Multi-machine (exo) note

These artifacts fit on one machine, but if you shard them across an
[exo](https://github.com/exo-explore/exo) cluster anyway, one guard is
required: VQ codebooks must **replicate rather than slice**. Stock exo
tensor parallelism slices them. This artifact's bundled `model.py` detects
that and fails loudly with an explanatory error (instead of silently
generating fluent garbage that reads as "a broken quant") — but it cannot
fix the sharding itself. To actually run tensor-parallel, apply
[exo PR #2268](https://github.com/exo-explore/exo/pull/2268) or run the
ready branch
[`noahzelezny/exo:vq-codebook-replicate`](https://github.com/noahzelezny/exo/tree/vq-codebook-replicate).
Single-machine mlx-lm and pipeline sharding are unaffected.

## Limitations

- Measurably below the 18.7 GiB sibling (1.029x vs 0.991x ppl) and well
  below the 35 GiB 8-bit. This is the size-first choice, not the quality
  choice.
- Perplexity measured on one corpus; a different workload may rank builds
  differently.
- No blind human-preference evaluation was run on this model.

## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)
## Provenance

Base model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — **Apache-2.0**.
This is a quantized derivative and inherits that licence; using it
means accepting the base model's terms.
Quantization: TheDrainFlorist, 2026.

Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab).
