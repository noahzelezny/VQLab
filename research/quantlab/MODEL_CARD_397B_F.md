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

# Qwen3.5-397B-A17B-VQ-2.2bpw

**101.0 GiB — the accessibility build.**

**v2 — updated 2026-08-22.** This repository now serves a rebuilt artifact at
the same size and the *same bits per weight*, with a different codebook
geometry that measures better on both perplexity corpora. v1's numbers are
kept below rather than quietly overwritten, and **v1's bytes remain
downloadable** by pinning the previous revision:

```python
snapshot_download("TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw",
                  revision="4554635165011f67e8166fd94d4bcc8cbf91401c")  # v1
```

A vector-quantized build of [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
built to answer one question: **how small can a 397B get and still be worth
running?** 101.0 GiB — it runs on a single 128 GB Apple Silicon machine with
~17 GiB more headroom than our `VQ-2.4bpw` build, no cluster, no patches,
stock `mlx-lm`.

## Measured results

All numbers measured on this exact artifact and scored twice to an identical
total negative log-likelihood, with an unmodified `mlx-lm` install. Read the
whole row, not one cell:

| | **this model, v2** (101.0 GiB) | v1 (100.9 GiB) | spicyneuron 2.6bit (120.6 GiB) | `VQ-2.4bpw` (111.6 GiB) |
|---|---|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **3.0591** | 3.1706 | 3.1843 | 2.7655 |
| code perplexity (mixed-language) | **2.6728** | 2.6988 | 2.6667 | 2.6383 |

**The honest trade:** v2 improves on v1 by 3.5% on prose and 1.0% on code at
the same size and the same bits per weight. Against the closest community
quant it is better on prose (−3.9%) and level on code (2.6728 vs 2.6667 —
0.2%, which we do not claim in either direction; no fit-to-fit noise floor
has been measured at this geometry), at **19.6 GiB smaller**. Against
`VQ-2.4bpw` it still gives up real quality on both corpora.

**What the size buys.** A "128 GB" machine has **119.2 GiB** of usable memory
(vendors count in decimal GB; memory is allocated in binary GiB):

| build | resident at 8k context | free for KV cache, OS, everything else |
|---|---|---|
| this build | ~101.8 GiB | **~17.4 GiB** |
| `VQ-2.4bpw` | ~112.5 GiB | ~6.7 GiB |

That is the reason this build exists. `VQ-2.4bpw` is the better model and it
fits a 128 GB machine with very little room to spare. If your machine runs
`VQ-2.4bpw` comfortably at the context you need, use that one.

**Speed.** v2's 16,384-entry codebook no longer fits in Metal threadgroup
memory, so it reads the codebook from device memory and runs roughly **20%
slower than v1**. We are not publishing throughput figures: repeat runs of the
same artifact on the same machine varied more than the effect we would be
reporting, and an unreliable number is worse than none. Measure on your own
hardware.

Note the size does **not** buy speed in any case — this is an A17B MoE, so
decode reads the same active experts per token as the larger builds. It buys
*residency*, which is the table above.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw \
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
**codebook of joint 8-weight patterns** and stores one index per group.
Each 8-weight subvector stores one **14-bit index** into a per-tensor 16,384-entry fp16 codebook. At the same bits, the codebook's entries sit
where the weight distribution actually is, rather than on a uniform lattice —
which is why this beats scalar quantization at matched size rather than
merely matching it. Per-tensor codebooks, with an fp16 scale per (row, 64
weights), for 2.00 bits/weight stored in the expert region.

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
(prefix-8192) and a mixed-language code corpus — every number scored twice to
an identical total negative log-likelihood, with an unmodified `mlx-lm`. Two
corpora
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
| **Qwen3.5-397B-A17B-VQ-2.2bpw** *(v1 weights)* | 100.9 GiB | 0.861 | 0.841 | 0.787 |
| Qwen3.5-397B-A17B-VQ-2.4bpw | 111.6 GiB | 0.883 | 0.844 | 0.784 |
| spicyneuron 2.6bit | 120.6 GiB | 0.880 | 0.841 | 0.771 |
| Qwen3.5-397B-A17B-VQ-3.1bpw | 143.7 GiB | 0.903 | 0.840 | 0.780 |
| spicyneuron 3.5bit | 165.6 GiB | 0.904 | 0.846 | 0.767 |

Every model scored identical items, so differences are **paired** (McNemar
exact test). HellaSwag reliably separates these quants and reproduces the
perplexity ordering; PIQA and WinoGrande separate no pair at n=1000 and
stand as integrity checks rather than rankings.

These rows were measured on **v1's weights**; v2 has not been re-evaluated on
this harness, and the row above is labelled accordingly rather than reused for
different weights. v2 improves on v1 on both perplexity corpora, but that is
not a task-suite result and is not presented as one.

**v1** was statistically indistinguishable from every larger model
here on PIQA and WinoGrande; on HellaSwag it trails them by 2–4 points
(paired p < 0.02) — the measured cost of the smallest size in this
comparison. It is the accessibility artifact: the one that runs on a 128 GB
Mac with real headroom.

> These are **0-shot** scores. Leaderboard conventions often use 10-shot
> HellaSwag / 5-shot WinoGrande, which run several points higher — compare
> against other 0-shot numbers only.

## Tuning: prefill speed

`mlx-lm` prefills prompts in 512-token steps by default. This model is a
sparse MoE — a larger step puts more rows through each expert per call, which
this quantization format likes. **The throughput figures below were measured
on v1**; the knob and the memory costs apply unchanged to v2, the absolute
rates will be somewhat lower. Measured on an M4 Max 128 GB, 8k context:

| `--prefill-step-size` | prefill tok/s | peak memory |
|---|---|---|
| 512 (default) | ~63–76 | 101.8 GiB |
| 1024 | ~89 | 103.0 GiB |
| 2048 | ~112–118 | 105.4 GiB |
| 4096 | ~138–141 | 109.6 GiB |

Decode is unaffected (~18–21 tok/s throughout) — it uses a different code
path. The cost is memory: budget the peak above plus your KV cache. On a
128 GB machine this model has room for step 4096; leave headroom if you run
long contexts. (Measured single-box; with more memory or an exo cluster the
same knob applies with a higher ceiling.)

Note: perplexity is deterministic; wall-time figures are not, and will vary
with whatever else your machine is doing.

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

- Code-heavy workloads measurably prefer the `VQ-2.4bpw` build (+1.2% code
  perplexity here vs the community 2.6bit, +2.3% vs our `VQ-2.4bpw` build).
- This is a *thinking* model (Qwen3.5 family): it spends tokens reasoning
  before answering. Budget `max_tokens` accordingly.
- Distributed (exo) tensor-parallel serving needs one guard in exo's own
  sharding rule — VQ codebooks are a shared lookup table and must be
  replicated, not sliced. Submitted upstream as [PR #2268](https://github.com/exo-explore/exo/pull/2268); until it
  merges, a ready-to-use branch is at
  [`noahzelezny/exo:vq-codebook-replicate`](https://github.com/noahzelezny/exo/tree/vq-codebook-replicate) (8 lines in
  `src/exo/worker/engines/mlx/auto_parallel.py`, plus builtin model cards for
  this lineup). `mlx-lm` itself is stock in that setup too: verified serving
  this model across two Macs with an unpatched `mlx-lm`, producing output
  identical to the patched run. Single-box users are unaffected.

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
