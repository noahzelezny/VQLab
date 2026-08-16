---
language: en
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-397B-A17B
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
---

# Qwen3.5-397B-A17B-VQ-2.2bpw

**100.1 GiB — the accessibility build.**

A vector-quantized build of [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
built to answer one question: **how small can a 397B get and still be worth
running?** 100.1 GiB — it runs on a single 128 GB Apple Silicon machine with
~17 GiB more headroom than our `VQ-2.4bpw` build, no cluster, no patches,
stock `mlx-lm`.

## Measured results

All numbers measured on this exact artifact, reproduced bit-identically ×2,
scored with an unmodified `mlx-lm` install. Read the whole row, not one cell:

| | this model (100.1 GiB) | spicyneuron 2.6bit (120.6 GiB) | our `VQ-2.4bpw` build |
|---|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **3.1706** | 3.1843 | 2.7655 |
| code perplexity (mixed-language) | 2.6988 | **2.6667** | 2.6383 |

**The honest trade:** against the closest community quant this build is
slightly better on prose (−0.43%) and slightly worse on code (+1.20%), at
**20.5 GiB smaller**. Against our `VQ-2.4bpw` build it gives up real
quality on both corpora in exchange for ~10.7 GiB of headroom. If your
machine runs the `VQ-2.4bpw` build comfortably, use that one; this build
exists for machines and workloads where those gigabytes decide whether the
model fits at all.

Runtime, single M4 Max 128 GB (macOS, stock `mlx-lm`):

| | |
|---|---|
| load time | ~116 s (network volume; local SSD is faster) |
| resident memory | 100.1 GiB (**peak 103.0 GiB at 8k context**) |
| decode | **~20–21 tok/s**, flat from 512 → 8k context |
| prefill | ~42–48 tok/s (chunked, as mlx-lm does natively) |

Note the size does **not** buy speed — this is an A17B MoE, so decode reads
the same active experts per token as the larger builds. It buys *residency*:
~25 GiB of free memory on a 128 GB box while serving 8k context.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate --model <this-folder> \
  --prompt "Explain vector quantization briefly." --max-tokens 200
```

No patches, no forks: `config.json` declares `model_file: model.py`, and
`mlx-lm` imports the bundled `model.py` from inside this folder. That file
carries the VQ runtime (JIT-compiled Metal kernels via
`mx.fast.metal_kernel`), including the sub-byte packed-code reader.

## Format

Expert weights are product-quantized: each 4-weight subvector stores one
**7-bit index** into a per-tensor 128-entry fp16 codebook, with an fp16
scale per (row, 64 weights). The 7-bit codes are **bit-packed** into
uint32 words (row-local, 32-code blocks) — 2.00 bits/weight stored, which
is what puts this build under 101 GiB. Packing is a pure representation
change: this artifact's perplexities are identical, to four decimals of
total negative log-likelihood, to its unpacked twin. Non-expert structure,
attention, a promoted layer tail, and routers keep their original
higher-precision quantization.

Codebooks are fit in pure weight space (k-means; no Hessian, no
activation data). Fit: ~26 min on one M3 Ultra; packing: ~15 min.

## Vision

The artifact includes the full 333-tensor vision tower at source precision
(0.85 GiB). `mlx-lm` is text-only for this architecture and ignores it;
[exo](https://github.com/exo-explore/exo) loads it from this folder
directly. `mlx-vlm` support requires its `model_file` loader hook
(upstream PR in progress).

## Known limitations

- Code-heavy workloads measurably prefer the `VQ-2.4bpw` build (+1.2% code
  perplexity here vs the community 2.6bit, +2.3% vs our `VQ-2.4bpw` build).
- This is a *thinking* model (Qwen3.5 family): it spends tokens reasoning
  before answering. Budget `max_tokens` accordingly.
- Distributed (exo) tensor-parallel serving needs a one-line sharding rule
  — VQ codebooks must be replicated, not sliced. Single-box users are
  unaffected.

## Provenance

Base model: Qwen/Qwen3.5-397B-A17B (Apache 2.0 — see the base model card
for license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX;
referee scoring scripts and the full experiment log (what worked, what was
falsified, and why) available on request.
