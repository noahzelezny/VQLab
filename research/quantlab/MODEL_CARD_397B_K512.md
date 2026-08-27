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

# Qwen3.5-397B-A17B-VQ-2.6bpw

**122.3 GiB — the pound-for-pound build.** It lands in the same size class
as the strongest community quant at this rate; the table below is the
comparison. A vector-quantized build of
[Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) for Apple
Silicon. Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint as `model.py`. It sits between the `VQ-2.4bpw` daily driver and the
`VQ-3.1bpw` quality build, and needs the same hardware class as the latter.

## Measured results

Scored on this exact artifact with an unmodified `mlx-lm` install, on the same
two corpora and the same harness as every comparator below:

| | this model (122.3 GiB) | spicyneuron 2.6bit (120.6 GiB) |
|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.5634** | 3.1843 |
| code perplexity (mixed-language) | **2.6123** | 2.6667 |

**19.5% better on prose and 2.0% better on code, at 1.7 GiB more on disk.**

Both margins are large relative to the fit-to-fit noise we can measure: 24x
and 3.1x respectively. One caveat on those multiples, because it is the kind
of thing that is easy to leave out — the noise floor they are quoted against
was measured at a *different* codebook size (256, not 512). No floor has been
measured at this geometry. Floors in this project have widened every time
they were measured more carefully, so treat 24x as "comfortably real" rather
than as a precise figure. The prose gap is not in doubt at any plausible
floor; the code gap is the smaller of the two and would be the first to
become uninteresting if this geometry's floor turned out to be wide.

**The size comparison understates us, and here is by how much.** This
artifact carries the full vision tower at bf16 (333 tensors, 0.849 GiB). The
spicyneuron builds are text-only and carry none. Comparing text weights to
text weights, this build is 121.5 GiB against their 120.6 — **+0.88 GiB, not
+1.7.** We quote the download figure above because that is what you actually
fetch, but the like-for-like number is the smaller one.

## Hardware

**This build does not fit a 128 GB machine.** You need either:

- a single Apple Silicon machine with ≥ 192 GB unified memory, or
- an [exo](https://github.com/exo-explore/exo) cluster (verified on 96 GB +
  128 GB over Thunderbolt) with one sharding guard: VQ codebooks must
  replicate rather than slice. Submitted upstream as
  [PR #2268](https://github.com/exo-explore/exo/pull/2268); until it merges,
  use [`noahzelezny/exo:vq-codebook-replicate`](https://github.com/noahzelezny/exo/tree/vq-codebook-replicate).

Verified serving on a 2-node ring: placed and serving in 99 seconds,
800-token coherent generation, and three graded known-answer probes returned
correct. **No throughput figure is published here** — we measured placement
and correctness, not tokens per second, and an unmeasured number is worse
than none.

Because it exceeds a single 128 GB box, this artifact cannot be verified
single-node. Any re-verification has to be a 2-node ring.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.6bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 1000
```

`max-tokens` is deliberately generous: this is a reasoning model and a small
budget gets consumed by its thinking, leaving the visible answer truncated.

No patches, no forks: `config.json` declares `model_file: model.py`, and
`mlx-lm` imports the bundled `model.py` from inside this folder. That file
carries the VQ runtime — JIT-compiled Metal kernels via
`mx.fast.metal_kernel` — including the sub-byte packed-code reader.

## Methodology

**Mixed precision by layer sensitivity.** Attention, MoE routers, embeddings
and the output head stay at higher precision — a small fraction of the
parameters, but errors there propagate through every token. The MoE *experts*
are ~85% of the model and individually far more tolerant, so they absorb the
aggressive quantization.

**Vector quantization instead of scalar rounding — the part that is
different.** Scalar 2-bit gives each
weight 4 rigid levels; over a group of 4 weights that is 256 fixed grid
combinations. This build learns a **codebook of joint 4-weight patterns** and
stores one index per group. Each 4-weight subvector stores one **9-bit index**
into a per-tensor 512-entry fp16 codebook, with an fp16 scale per (row, 64
weights) — 2.5 bits/weight stored in the expert region. At the same bits the
codebook's entries sit where the weight distribution actually is, rather than
on a uniform lattice.

Every expert tensor uses this one geometry; there is no mixed allocation and
no per-layer schedule. Flat rungs are the reference points in this lineup
because no mixed-allocation build we measured beat the flat rung at its own
size.

**Codebooks are fit in pure weight space** — k-means over the weight
subvectors, no Hessian, no activation statistics, no calibration corpus.
Calibration-fitted methods we tested (GPTQ- and DWQ-style) reduced *layer*
error while making *end-to-end* perplexity worse on this architecture, and
they bias the result toward whatever text the calibration set contains.

**The fit is not seeded.** k-means draws an unseeded subsample, so this
artifact is reproducible in recipe and geometry but not bit-for-bit. That is
why margins here are quoted against a measured fit-to-fit floor rather than
against a repeated build.

**Sub-byte bit-packing.** Codes are packed into uint32 words (row-local,
32-code blocks) rather than padded to whole bytes, which is what makes the
non-byte-aligned size possible. Packing is a pure representation change.

**How it was evaluated.** Perplexity on two corpora — raw wikitext
(prefix-8192) and a mixed-language code corpus — scored with an unmodified
`mlx-lm` on the same harness used for every comparator here. Two corpora
because this family shows real domain asymmetry: larger codebooks buy far
more on prose than on code, so a single-corpus number would misrepresent the
trade.

## Task benchmarks

**Not yet measured on this artifact.** Two of the siblings above carry
HellaSwag/PIQA/WinoGrande numbers; this build has not been run through that
harness, and reporting a sibling's task scores here would be exactly the
substitution this project refuses to make. They will be added once the suite
has been run under the same harness (lm-eval 0.4.12, layer-streaming
loglikelihood scorer, 0-shot, first 1000 items per task).

## Vision

The artifact includes the full 333-tensor vision tower at source precision
(0.85 GiB). `mlx-lm` is text-only for this architecture and ignores it;
[exo](https://github.com/exo-explore/exo) loads it from this folder
directly. `mlx-vlm` support requires its `model_file` loader hook
([PR #1926](https://github.com/Blaizzy/mlx-vlm/pull/1926), under review).

The sizes quoted above are the download: they include this tower. Because
`mlx-lm` does not load it, resident memory runs ~0.85 GiB below the disk
figure.

## Siblings

All from the same skeleton and recipe, all scored the same way:

| | size | wikitext | code | needs |
|---|---|---|---|---|
| `VQ-2.2bpw` | 101.0 GiB | 3.0591 | 2.6728 | 128 GB Mac, roomy |
| `VQ-2.4bpw` | 111.6 GiB | 2.7655 | 2.6383 | 128 GB Mac, tight |
| **`VQ-2.6bpw` (this build)** | **122.3 GiB** | **2.5634** | **2.6123** | ≥192 GB or cluster |
| `VQ-3.1bpw` | 143.7 GiB | 2.3410 | 2.5963 | ≥192 GB or cluster |

If you can run this build you can run `VQ-3.1bpw`, which is better on both
corpora for 21.4 GiB more. Take this one if those gigabytes are worth more to
you than the quality difference — on a 192 GB machine it leaves roughly 70 GiB
free against the 3bpw build's 48.

## Known limitations

- No throughput measurement — see Hardware.
- Perplexities are corpus-specific. Compare only against models scored on the
  same files, never across harnesses.
- This is a *thinking* model: it spends tokens reasoning before answering.
  Budget `max_tokens` accordingly.

## Acknowledgment

spicyneuron's 397B quants are what made this model runnable on my hardware in
the first place, and they were the reference this work was measured against
throughout. This release is offered in that spirit: the method, the failures
as well as the wins, and comparator numbers re-measured on one harness so the
claims can be checked rather than taken on trust.

## Paper

The method, the full three-model ladder, the negative results, and the
measurement rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119018) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

## Provenance

Base model: Qwen/Qwen3.5-397B-A17B (Apache 2.0 — see the base model card for
license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX.
