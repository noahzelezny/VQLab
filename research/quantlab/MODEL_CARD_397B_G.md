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

All numbers measured on this exact artifact, scored with an unmodified
`mlx-lm` install on one harness:

| | this model (143.7 GiB) | spicyneuron 3.5bit (165.6 GiB) | our previous `VQ-3.1bpw` (143.7 GiB) |
|---|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.3410** | 2.3614 | 2.3519 |
| code perplexity (mixed-language) | **2.5963** | 2.6005 | 2.5987 |

**The honest claim: 21.9 GiB smaller than the community 3.5bit, better on
wikitext, tied on code.** The wikitext margin (-0.86%) is 3.6x this
geometry's measured fit-to-fit noise floor, so it is a result. The code
margin (-0.16%) is 0.4x that floor — inside the range two independent fits
of the same recipe produce — so we report it as a tie rather than a win.

**What changed against our own previous build at this size.** This artifact
replaces `VQ-3.1bpw`. It is the **same size and same geometry** — 143.682 GiB,
flat d4/K2048 experts, an identical footprint — and it measures **−0.46%
wikitext and −0.09% code** against it. Neither difference is a claim: the
code gap is 0.2x and the wikitext gap 1.9x the fit-to-fit noise floor we
later measured at this geometry, and our bar for calling a single-fit result
is 3x. The two artifacts are best described as equivalent, with this one
carrying the cleaner provenance.

**Why there is no story here.** We spent weeks treating that gap as an effect
with a cause — the leading theory being that the newer codebooks came from a
later version of our k-means implementation. Testing it properly required
something we had never measured: how much two runs of the *same* recipe differ
from each other. Our codebook fitter draws an unseeded subsample, so it does
not produce the same codebooks twice. When we finally measured that spread at
this geometry it came to 0.0056 wikitext perplexity, which is most of the gap
we had been trying to explain. There is no mechanism to report because there
is no longer an effect large enough to need one.

**What survives is the measurement, which is what the table above reports.**
If you are choosing between this artifact and the one it replaces, take this
one for provenance rather than for the numbers — they are the same model to
within the precision the method can deliver.

**The part worth carrying away, if you fit your own codebooks,** comes out of
the same investigation and does not depend on its unresolved half: we spent
weeks comparing fits by *mean* reconstruction error, and across the body
tensors where these builds differ most, that mean moves by −0.00033 — it gets
slightly better while the model gets worse. Any fit gated on mean
reconstruction error is blind to that trade. It is how a regression survived
our own review, and it is the one lesson here we are confident transfers.

This repository was previously published as `VQ-3.1bpw`; it has been renamed
and `main` now serves the improved weights, so the old URL redirects here.
**The previous build is preserved, not deleted** — it remains fetchable at
the revision it was published at:

```bash
huggingface-cli download TheDrainFlorist/Qwen3.5-397B-A17B-VQ-3.1bpw \
  --revision a0da72a0c43932704a272fe3ce6a6513194570eb
```

So the row above is not an orphaned claim: the artifact that produced it is
still downloadable and the comparison stays checkable against the same
harness.

The `3bpw` label is also a correction: both artifacts compute to **3.045
bits/weight** over text weights (3.06 including the vision tower), so the
earlier `3.1` overstated by ~0.05. The size did not change.

Domain asymmetry note: against our smaller builds, the wikitext gain is
much larger than the code gain (codebook size buys prose more than code on
this family). Judge by your workload; never compare perplexities across
different corpora or eval harnesses.

## Hardware

143.7 GiB on disk (~142.8 GiB resident) does **not** fit a 128 GB machine. You need either:

- a single Apple Silicon machine with ≥ 192 GB unified memory, or
- an [exo](https://github.com/exo-explore/exo) cluster (e.g. 96 GB + 128 GB
  over Thunderbolt) with one sharding guard: VQ codebooks replicate rather
  than slice.

**Multi-box sharding requires our fork.** Stock exo tensor-parallel slices
the VQ codebooks, which produces fluent nonsense rather than an error — the
model appears to load and run normally. The guard was submitted upstream as
[PR #2268](https://github.com/exo-explore/exo/pull/2268); **it has not been
merged and has seen no upstream movement**, so until it does, use the branch
[`noahzelezny/exo:vq-codebook-replicate`](https://github.com/noahzelezny/exo/tree/vq-codebook-replicate),
which also ships builtin exo model cards for this lineup so the model is
runnable by name. This artifact's bundled `model.py` additionally carries a
runtime guard that raises loudly if a sliced codebook reaches the forward
path, so a misconfigured cluster fails instead of lying.

Cluster throughput for this artifact: **not yet measured — no number is
claimed here.**

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
(prefix-8192) and a mixed-language code corpus — scored with an unmodified
`mlx-lm` on the same harness used for every comparator here. Two corpora
because this family shows real domain asymmetry: larger codebooks buy far
more on prose than on code, so a single-corpus number would misrepresent the
trade. Task-suite results for this artifact are not yet available — see
below.

## Task benchmarks

**Not yet re-measured on this artifact.** The task-suite table published on
the previous build was measured on *that* artifact; although this one is the
same size and geometry, its codebooks are different, and reporting another
artifact's task scores here would be exactly the substitution this project
refuses to make. They will be added once the suite has been re-run under the
same harness (lm-eval 0.4.12, layer-streaming loglikelihood scorer, 0-shot,
first 1000 items per task).

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

All from the same skeleton and recipe, all scored the same way:

| | size | wikitext | code | needs |
|---|---|---|---|---|
| `VQ-2.2bpw` | 101.0 GiB | 3.0591 | 2.6728 | 128 GB Mac, roomy |
| `VQ-2.4bpw` | 111.6 GiB | 2.7655 | 2.6383 | 128 GB Mac, tight |
| `VQ-2.6bpw` | 122.3 GiB | 2.5634 | 2.6123 | ≥192 GB or cluster |
| **`VQ-3.1bpw` (this build)** | **143.7 GiB** | **2.3410** | **2.5963** | ≥192 GB or cluster |

This is the quality end of the ladder: best on both corpora, and the largest.
If it does not fit, `VQ-2.6bpw` is 21.4 GiB smaller and the next step down on
both corpora; `VQ-2.4bpw` is the largest that runs comfortably on a 128 GB
machine.

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

## Paper

The method, the full three-model ladder, the negative results, and the
measurement rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119018) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

## Provenance

Base model: Qwen/Qwen3.5-397B-A17B (Apache 2.0 — see the base model card
for license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX;
referee scoring scripts and the full experiment log (what worked, what was
falsified, and why) available on request.
