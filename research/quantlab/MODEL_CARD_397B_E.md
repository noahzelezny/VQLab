---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-397B-A17B
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- qwen3.5
---

# Qwen3.5-397B-A17B-VQ-3.1bpw

**143.7 GiB — the quality build.**

A vector-quantized build of [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
for machines with memory to spend: the strongest quantization we know how
to make of this model at this size, on stock `mlx-lm`, no patches.

## Measured results

All numbers measured on this exact artifact, reproduced bit-identically ×2,
scored with an unmodified `mlx-lm` install:

| | this model (143.7 GiB) | spicyneuron 3.5bit (165.6 GiB) | our `VQ-2.4bpw` build |
|---|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.3519** | 2.3614 | 2.7655 |
| code perplexity (mixed-language) | 2.5987 | 2.6005 | 2.6383 |

**The honest claim: matches the community 3.5bit on code (0.07% is a tie),
edges it on wikitext (−0.40%), at 21.9 GiB smaller.** The size difference
is the story — this quality previously cost 165.6 GiB.

Domain asymmetry note: against our smaller builds, the wikitext gain is
much larger than the code gain (codebook size buys prose more than code on
this family). Judge by your workload; never compare perplexities across
different corpora or eval harnesses.

## Hardware

143.7 GiB on disk (~142.8 GiB resident) does **not** fit a 128 GB machine. You need either:

- a single Apple Silicon machine with ≥ 192 GB unified memory, or
- an [exo](https://github.com/exo-explore/exo) cluster (e.g. 96 GB + 128 GB
  over Thunderbolt) with one sharding rule: VQ codebooks replicate rather
  than slice — a one-line change to exo's `auto_parallel`, not yet upstreamed.

Measured on the cluster (exo, M3 Ultra 96 GB + M4 Max 128 GB over
Thunderbolt 5 / RDMA, tensor-sharded): **~17.4 tok/s** decode on a short
prompt, placing in ~5.5 minutes. No single-box number — we have no machine
with ≥192 GB unified memory to measure one; for reference the `VQ-2.4bpw`
build decodes at 19–22 tok/s on one M4 Max, so a single large-memory box
should land in that class without the ring's communication overhead.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.5-397B-A17B-VQ-3.1bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 1000
```

`max-tokens` is deliberately generous: this is a reasoning model and a small
budget gets consumed by its thinking, leaving the visible answer truncated.

No patches, no forks: `config.json` declares `model_file: model.py`, and
`mlx-lm` imports the bundled `model.py` from inside this folder. That file
carries the VQ runtime (JIT-compiled Metal kernels via
`mx.fast.metal_kernel`), including the sub-byte packed-code reader.

## Methodology

**Mixed precision by layer sensitivity.** Not all weights deserve the same
bits. Attention, MoE routers, embeddings, and the output head stay at higher
precision — they are a small fraction of the parameters but errors there
propagate through every token. The MoE *experts* are ~85% of the model and
individually far more tolerant, so they absorb the aggressive quantization.
A tail of later layers is also promoted above the expert baseline; measured
layer-wise error showed depth matters, and the last layers repay the bits.

**Vector quantization instead of scalar rounding — the part that is
different.** Scalar 2-bit gives each weight 4 rigid levels; over a group of 4
weights that is 256 fixed grid combinations. This build instead learns a
**codebook of joint 4-weight patterns** and stores one index per group.
Each 4-weight subvector stores one **11-bit index** into a per-tensor 2048-entry fp16 codebook. At the same bits, the codebook's entries sit
where the weight distribution actually is, rather than on a uniform lattice —
which is why this beats scalar quantization at matched size rather than
merely matching it. Per-tensor codebooks, with an fp16 scale per (row, 64
weights), for 3.00 bits/weight stored in the expert region.

**Codebooks are fit in pure weight space** — k-means over the weight
subvectors, no Hessian, no activation statistics, no calibration corpus. That
is a deliberate choice: calibration-fitted methods we tested (GPTQ- and
DWQ-style) reduced *layer* error while making *end-to-end* perplexity worse
on this architecture, and they bias the result toward whatever text the
calibration set contains. Weight-space fitting has no such domain preference.

**Sub-byte bit-packing.** Codes are packed into uint32 words (row-local,
32-code blocks) rather than padded to whole bytes, which is what makes the
non-byte-aligned sizes possible at all. Packing is a pure representation
change: the packed artifact's perplexities match its unpacked twin to four
decimals of total negative log-likelihood on both corpora.

**How it was evaluated.** Perplexity on two corpora — raw wikitext
(prefix-8192) and a mixed-language code corpus — every number reproduced
bit-identically twice, scored with an unmodified `mlx-lm`. Two corpora
because this family shows real domain asymmetry: larger codebooks buy far
more on prose than on code, so a single-corpus number would misrepresent the
trade. Task-suite results are reported below, measured the same way.

## Task benchmarks

All five models below — this repo's three VQ artifacts and the two community
comparators — were evaluated on the **same harness, same settings, same
seeded items**: lm-eval 0.4.12 driven by a layer-streaming loglikelihood
scorer (`mlx-lm` 0.31.3), **0-shot**, first 1000 items per task, `acc_norm`
for HellaSwag/PIQA, `acc` for WinoGrande. Task numbers published elsewhere
come from a different pipeline and are not directly comparable, so the
comparator *artifacts* were re-evaluated here under identical conditions
rather than quoting their reported figures.

| model | size | HellaSwag | PIQA | WinoGrande |
|---|---|---|---|---|
| Qwen3.5-397B-A17B-VQ-2.2bpw | 100.9 GiB | 0.861 | 0.841 | 0.787 |
| Qwen3.5-397B-A17B-VQ-2.4bpw | 111.6 GiB | 0.883 | 0.844 | 0.784 |
| spicyneuron 2.6bit | 120.6 GiB | 0.880 | 0.841 | 0.771 |
| **Qwen3.5-397B-A17B-VQ-3.1bpw** *(this model)* | 143.7 GiB | 0.903 | 0.840 | 0.780 |
| spicyneuron 3.5bit | 165.6 GiB | 0.904 | 0.846 | 0.767 |

Every model scored identical items, so differences are **paired** (McNemar
exact test). HellaSwag reliably separates these quants and reproduces the
perplexity ordering; PIQA and WinoGrande separate no pair at n=1000 and
stand as integrity checks rather than rankings.

**This model** is statistically indistinguishable from spicyneuron's
3.5bit on all three tasks (McNemar p = 1.00 / 0.33 / 0.25) at **21.9 GiB
smaller**, and matches it on perplexity for both corpora — the same quality
point, one memory class earlier.

> These are **0-shot** scores. Leaderboard conventions often use 10-shot
> HellaSwag / 5-shot WinoGrande, which run several points higher — compare
> against other 0-shot numbers only.

## Vision

The artifact includes the full 333-tensor vision tower at source precision
(0.85 GiB). `mlx-lm` is text-only for this architecture and ignores it;
[exo](https://github.com/exo-explore/exo) loads it from this folder
directly. `mlx-vlm` support requires its `model_file` loader hook
([PR #1926](https://github.com/Blaizzy/mlx-vlm/pull/1926), under review).

The sizes quoted above are the download: they include this tower. Because
`mlx-lm` does not load it, resident memory runs ~0.85 GiB below the disk
figure — the runtime tables report what was actually measured resident.

## Known limitations

- Needs ≥ 192 GB unified memory or a cluster — see Hardware above.
- This is a *thinking* model (Qwen3.5 family): it spends tokens reasoning
  before answering. Budget `max_tokens` accordingly.

## Acknowledgment

spicyneuron's 397B quants are what made this model runnable on my hardware in
the first place — they were the artifacts that fit when nothing else did, and
they were the reference this work was measured against throughout. This
release is offered in that same spirit: the full method, the experiments that
failed as well as the ones that worked, and comparator numbers re-measured on
one harness so the claims can be checked rather than taken on trust.

## Provenance

Base model: Qwen/Qwen3.5-397B-A17B (Apache 2.0 — see the base model card
for license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX;
referee scoring scripts and the full experiment log (what worked, what was
falsified, and why) available on request.
