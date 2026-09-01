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

# Qwen3.5-397B-A17B-VQ-2.4bpw

**111.6 GiB — the daily driver, runs on a single 128 GB Mac.**

A vector-quantized build of [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
that fits and **generates on one 128 GB Apple Silicon machine** — no cluster,
no patches, stock `mlx-lm`.

## Measured results

All numbers measured on this exact artifact (not projected from a proxy) and
scored twice to an identical total negative log-likelihood, with an
unmodified `mlx-lm` install.

| | this model (111.6 GiB) | spicyneuron 2.6bit (120.6 GiB) |
|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.7655** | 3.1843 |
| code perplexity (mixed-language) | 2.6383 | 2.6667 |

Runtime, single M4 Max 128 GB (macOS, stock `mlx-lm`):

| | |
|---|---|
| load time | ~60 s |
| resident memory | 110.8 GiB (peak 117.7 GiB at 30k context) |
| context verified | **30,031 tokens**, zero swap growth |
| decode | **~19–21 tok/s**, flat from 512 → 14k context |
| prefill | **~79 tok/s** at a 1024-token step (see Tuning — this model is memory-bound on a 128 GB box; bigger steps get SLOWER) |

Perplexities are corpus-specific: never compare them across different
corpora or eval harnesses, only against other models scored on the same
files. The wikitext margin (13.2%) is a real result — 16x this geometry's
measured fit-to-fit noise floor. The code margin (1.07%) is **not**: at 1.6x
the floor it is inside the range two independent fits of the same recipe
produce, so treat code as a tie and judge by your workload.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.4bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 1000
```

`max-tokens` is deliberately generous: this is a reasoning model and a small
budget gets consumed by its thinking, leaving the visible answer truncated.

No patches, no custom forks: `config.json` declares `model_file: model.py`,
and `mlx-lm` imports the bundled `model.py` from inside this folder. That
file carries the VQ runtime — JIT-compiled Metal kernels via
`mx.fast.metal_kernel` — so a stock install can read the format.

Tips for 128 GB machines:
- Close memory-heavy apps first; the model wants ~111 GiB resident and
  peaks ~118 GiB at long context.
- `VQ_DECODE_CHUNK` (env var) trades prefill speed for peak memory
  during long-prompt processing. The default auto-sizes from free memory;
  lower it (e.g. `16`) if you run close to the ceiling.
- Machines with more memory need none of this.

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
Each 4-weight subvector stores one **8-bit index** into a per-tensor 256-entry fp16 codebook. At the same bits, the codebook's entries sit
where the weight distribution actually is, rather than on a uniform lattice —
which is why this beats scalar quantization at matched size rather than
merely matching it. Per-tensor codebooks, with an fp16 scale per (row, 64
weights), for 2.25 bits/weight stored in the expert region.

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
| Qwen3.5-397B-A17B-VQ-2.2bpw *(v1 weights)* | 100.9 GiB | 0.861 | 0.841 | 0.787 |
| **Qwen3.5-397B-A17B-VQ-2.4bpw** *(this model)* | 111.6 GiB | 0.883 | 0.844 | 0.784 |
| spicyneuron 2.6bit | 120.6 GiB | 0.880 | 0.841 | 0.771 |
| `VQ-3.1bpw` *(predecessor weights at this name)* | 143.7 GiB | 0.903 | 0.840 | 0.780 |
| spicyneuron 3.5bit | 165.6 GiB | 0.904 | 0.846 | 0.767 |

Every model scored identical items, so differences are **paired** (McNemar
exact test). HellaSwag reliably separates these quants and reproduces the
perplexity ordering; PIQA and WinoGrande separate no pair at n=1000 and
stand as integrity checks rather than rankings.

**This model** is statistically indistinguishable from spicyneuron's
2.6bit on all three tasks (McNemar p = 0.76 / 0.77 / 0.29) at **9.0 GiB
smaller** — consistent with the perplexity result, where it leads its size
class on prose and ties on code.

> Two rows above were measured before their repos were updated: the 2.2bpw
> figure is v1 (d4/K128; the repo now serves d8/K16384), and the 143.7 GiB
> figure is the build that preceded the current `VQ-3.1bpw` weights. Both are
> labelled rather than deleted — they were measured on real artifacts, which
> remain fetchable at their published revisions. Re-measurement is queued.

> These are **0-shot** scores. Leaderboard conventions often use 10-shot
> HellaSwag / 5-shot WinoGrande, which run several points higher — compare
> against other 0-shot numbers only.

## Runtime update (2026-08-20)

`model.py` now dispatches this artifact's uint8 codes through the faster
fused kernel (a zero-copy view — the bytes are already in the packed-8
layout). Verified before publishing: greedy decode is **token-identical**
to the previous runtime on this artifact, and the perplexity above
reproduces to every decimal. No weights changed; re-download `model.py`
only.

Measured effect at 35B scale on an M3 Ultra: prompt processing +25–32%
(963–993 vs 769–772 tok/s at 2k/8k context), decode unchanged. **The
prefill numbers for THIS 397B model have not been re-measured with the new
runtime** — the table above reflects the previous runtime; treat any
speedup here as unverified until it is measured.

## Tuning: prefill speed

`mlx-lm` prefills prompts in 512-token steps by default, and a larger step
usually helps a sparse MoE. **On this model, only up to a point** — at
110.8 GiB it is the tightest fit of the three on a 128 GB machine, and memory
pressure reverses the gain. Measured on an M4 Max 128 GB, 8k context:

| `--prefill-step-size` | prefill tok/s | peak memory |
|---|---|---|
| 1024 | **~79** | 113.6 GiB |
| 2048 | ~68 (slower) | 116.0 GiB |
| 4096 | not recommended | exceeded available memory |

**On a 128 GB machine, do not raise the step past 2048 with this model** —
the 4096 run exhausted memory. If you want a larger prefill step on 128 GB,
use the 100.9 GiB sibling (`-VQ-2.2bpw`), which has the headroom for it and
reaches ~141 tok/s. These limits are about the single-box memory ceiling,
not the model: with more memory (192 GB+, or tensor-sharded across an exo
cluster) the larger steps are back on the table. Decode is unaffected
either way.

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

## Siblings

This is the second of a four-size family, all from the same skeleton and
recipe, all measured the same way:

| | size | wikitext | code | needs |
|---|---|---|---|---|
| `VQ-2.2bpw` (accessibility) | 101.0 GiB | 3.0591 | 2.6728 | 128 GB Mac, roomy |
| **`VQ-2.4bpw` (this build)** | **111.6 GiB** | **2.7655** | **2.6383** | 128 GB Mac, tight |
| `VQ-2.6bpw` | 122.3 GiB | 2.5634 | 2.6123 | ≥192 GB or cluster |
| `VQ-3.1bpw` (quality) | 143.7 GiB | 2.3410 | 2.5963 | ≥192 GB or cluster |

## Known limitations

- **Tight on 128 GB.** ~118 GiB peak against ~120 GiB usable leaves little
  room for other software. It runs; it is not roomy.
- This is a *thinking* model (Qwen3.5 family): by default it spends tokens
  reasoning before answering. Budget `max_tokens` accordingly.
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

## Paper

The method, the full three-model ladder, the negative results, and the
measurement rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

## Provenance

Base model: Qwen/Qwen3.5-397B-A17B (Apache 2.0 — see the base model card
for license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX;
referee scoring scripts and the full experiment log (what worked, what was
falsified, and why) available on request.
