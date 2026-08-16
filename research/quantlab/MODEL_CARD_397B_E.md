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

# Qwen3.5-397B-A17B — VQ 142.8 GiB (the quality build)

A vector-quantized build of [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
for machines with memory to spend: the strongest quantization we know how
to make of this model at this size, on stock `mlx-lm`, no patches.

## Measured results

All numbers measured on this exact artifact, reproduced bit-identically ×2,
scored with an unmodified `mlx-lm` install:

| | this model (142.8 GiB) | spicyneuron 3.5bit (165.6 GiB) | our 110.8 GiB build |
|---|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.3519** | 2.3614 | 2.7655 |
| code perplexity (mixed-language) | 2.5987 | 2.6005 | 2.6383 |

**The honest claim: matches the community 3.5bit on code (0.07% is a tie),
edges it on wikitext (−0.40%), at 22.8 GiB smaller.** The size difference
is the story — this quality previously cost 165.6 GiB.

Domain asymmetry note: against our smaller builds, the wikitext gain is
much larger than the code gain (codebook size buys prose more than code on
this family). Judge by your workload; never compare perplexities across
different corpora or eval harnesses.

## Hardware

142.8 GiB resident does **not** fit a 128 GB machine. You need either:

- a single Apple Silicon machine with ≥ 192 GB unified memory, or
- an [exo](https://github.com/exo-explore/exo) cluster (e.g. 96 GB + 128 GB
  over Thunderbolt) with one sharding rule: VQ codebooks replicate rather
  than slice (upstream PR pending; one-line change to `auto_parallel`).

We have not measured single-box throughput (no ≥192 GB machine in the lab).
The 110.8 GiB sibling decodes at 19–22 tok/s on an M4 Max, and this build
reads only ~33% more expert bytes per token — expect the same class of
speed, not a different experience.

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
**11-bit index** into a per-tensor 2048-entry fp16 codebook, with an fp16
scale per (row, 64 weights). The 11-bit codes are **bit-packed** into
uint32 words (row-local, 32-code blocks) — 3.00 bits/weight stored, which
is the difference between this artifact's 142.8 GiB and the 196.3 GiB it
would occupy unpacked. Packing is a pure representation change: this
artifact's perplexities are identical, to four decimals of total negative
log-likelihood, to its unpacked twin. Non-expert structure, attention, a
promoted layer tail, and routers keep their original higher-precision
quantization.

Codebooks are fit in pure weight space (k-means; no Hessian, no
activation data). Fit: ~4.3 h on one M3 Ultra; packing: ~20 min.

## Vision

The artifact includes the full 333-tensor vision tower at source precision
(0.85 GiB). `mlx-lm` is text-only for this architecture and ignores it;
[exo](https://github.com/exo-explore/exo) loads it from this folder
directly. `mlx-vlm` support requires its `model_file` loader hook
(upstream PR in progress).

## Known limitations

- Needs ≥ 192 GB unified memory or a cluster — see Hardware above.
- This is a *thinking* model (Qwen3.5 family): it spends tokens reasoning
  before answering. Budget `max_tokens` accordingly.

## Provenance

Base model: Qwen/Qwen3.5-397B-A17B (Apache 2.0 — see the base model card
for license and usage terms). Quantization: rotlab, 2026. Built with MLX;
referee scoring scripts and the full experiment log (what worked, what was
falsified, and why) available on request.
