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

# Qwen3.5-397B-A17B-VQ-2.4bpw

**110.8 GiB — the daily driver, runs on a single 128 GB Mac.**

A vector-quantized build of [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
that fits and **generates on one 128 GB Apple Silicon machine** — no cluster,
no patches, stock `mlx-lm`.

## Measured results

All numbers measured on this exact artifact (not projected from a proxy),
reproduced bit-identically ×2, scored with an unmodified `mlx-lm` install.

| | this model (110.8 GiB) | spicyneuron 2.6bit (120.6 GiB) |
|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.7655** | 3.1843 |
| code perplexity (mixed-language) | **2.6383** | 2.6667 |

Runtime, single M4 Max 128 GB (macOS, stock `mlx-lm`):

| | |
|---|---|
| load time | ~60 s |
| resident memory | 110.8 GiB (peak 117.7 GiB at 30k context) |
| context verified | **30,031 tokens**, zero swap growth |
| decode | **~19–22 tok/s**, flat from 512 → 14k context |
| prefill | ~40–50 tok/s (chunked, as mlx-lm does natively) |

Perplexities are corpus-specific: never compare them across different
corpora or eval harnesses, only against other models scored on the same
files. The wikitext margin (13.2%) is much larger than the code margin
(1.07%) — that asymmetry is real, so judge by your workload.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate --model <this-folder> \
  --prompt "Explain vector quantization briefly." --max-tokens 200
```

No patches, no custom forks: `config.json` declares `model_file: model.py`,
and `mlx-lm` imports the bundled `model.py` from inside this folder. That
file carries the VQ runtime — JIT-compiled Metal kernels via
`mx.fast.metal_kernel` — so a stock install can read the format.

Tips for 128 GB machines:
- Close memory-heavy apps first; the model wants ~111 GiB resident and
  peaks ~118 GiB at long context.
- `SCOUT_VQ_DECODE_CHUNK` (env var) trades prefill speed for peak memory
  during long-prompt processing. The default auto-sizes from free memory;
  lower it (e.g. `16`) if you run close to the ceiling.
- Machines with more memory need none of this.

## Format

Expert weights (the 2-bit region of a structure-quantized skeleton) are
product-quantized: each 4-weight subvector stores one uint8 index into a
per-tensor 256-entry fp16 codebook, with an fp16 scale per (row, 64
weights) — 2.25 bits/weight stored. Non-expert structure, attention, a
promoted layer tail, and routers keep their original higher-precision
quantization. Per-tensor geometry lives in `config.json → vq_modules`.

Codebooks are fit in pure weight space (k-means; no Hessian, no
activation data). Fit + assembly: ~2 h on one M4 Max.

## Vision

The artifact includes the full 333-tensor vision tower at source precision
(0.85 GiB). `mlx-lm` is text-only for this architecture and ignores it;
[exo](https://github.com/exo-explore/exo) loads it from this folder
directly. `mlx-vlm` support requires its `model_file` loader hook
(upstream PR in progress).

## Siblings

This is the middle of a three-size family, all from the same skeleton and
recipe, all measured the same way:

| | size | wikitext | code | needs |
|---|---|---|---|---|
| `VQ-2.2bpw` (accessibility) | 100.1 GiB | 3.1706 | 2.6988 | 128 GB Mac, roomy |
| **`VQ-2.4bpw` (this build)** | **110.8 GiB** | **2.7655** | **2.6383** | 128 GB Mac, tight |
| `VQ-3.1bpw` (quality) | 142.8 GiB | 2.3519 | 2.5987 | ≥192 GB or cluster |

## Known limitations

- **Tight on 128 GB.** ~118 GiB peak against ~120 GiB usable leaves little
  room for other software. It runs; it is not roomy.
- This is a *thinking* model (Qwen3.5 family): by default it spends tokens
  reasoning before answering. Budget `max_tokens` accordingly.
- Distributed (exo) tensor-parallel serving works but needs a one-line
  sharding rule — VQ codebooks must be replicated, not sliced. Single-box
  users are unaffected.

## Provenance

Base model: Qwen/Qwen3.5-397B-A17B (Apache 2.0 — see the base model card
for license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX;
referee scoring scripts and the full experiment log (E31–E36: what worked,
what was falsified, and why) available on request.
