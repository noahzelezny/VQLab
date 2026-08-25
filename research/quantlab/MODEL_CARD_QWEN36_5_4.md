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
---

# Qwen3.6-35B-A3B-VQ-5.4bpw

**22.2 GiB — the closest to bf16 in this collection.**

A vector-quantized build of
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) for Apple
Silicon. Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint as `model.py`.

## Measured results

Referee corpus, 2048-token windows, scored against the bf16 teacher on the
same files with an unmodified `mlx-lm`. All sizes include the bf16 vision
tower, which every build here carries:

| build | size | KL to bf16 (mnats/tok) | top-1 agreement |
|---|---|---|---|
| bf16 | 65.4 GiB | 0 | 100% |
| mlx-community 8-bit | 35.1 GiB | 7.4 | 96.18% |
| **this model** | **22.2 GiB** | **28.1** | **92.22%** |
| mlx-community 4-bit | 19.0 GiB | 78.6 | 85.61% |

**This build sits between the two community rungs and closer to the 8-bit
than to the 4-bit** — 28.1 millinats against their 78.6 and 7.4. Interpolating
the affine frontier between the 4-bit and a 6-bit build to this exact size
gives 38.7 millinats; this build measures 28.1, a factor of 1.4 below the
line.

**Two independent fits of this recipe were scored**, at 28.141 and 27.927
millinats — the only build in this collection measured twice. The spread
between them, 0.214 millinats, is this family's fit-to-fit noise floor, and
the margin above is roughly fifty times it. The number quoted in the table is
one artifact, the one published here, not an average of the two.

**Perplexity was not measured on this artifact.** The other releases in this
family carry a perplexity column; this build was scored on KL and top-1
agreement, which is the instrument this family is ranked by. Reporting a
sibling's perplexity here would be a substitution this project does not make.

## Runtime

**Not measured on this artifact.** No decode or memory benchmark has been run
on this build, and quoting a sibling's figures would be the same substitution.
Peak memory will run close to the disk figure less the vision tower, which
`mlx-lm` does not load.

Comfortable on a 32 GB machine.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.6-35B-A3B-VQ-5.4bpw \
  --prompt "Write a Python function that ..." \
  --max-tokens 512
```

## How it was built

Uniform vector quantization of the experts at **d=2, K=1024** — each
2-weight subvector stores one 10-bit index into a per-tensor 1024-entry fp16
codebook, with an fp16 scale per (row, 64 weights). Non-expert tensors are
8-bit.

Halving the subvector dimension is what buys the quality here. At d=4 every
additional quarter-bit costs a doubling of the codebook, so the d4 line
saturates; spending the same bits at d=2 keeps paying. There is no depth
schedule and no mixed allocation — every expert tensor uses the same
geometry.

**Codebooks are fit in pure weight space** — k-means over the weight
subvectors, no Hessian, no activation statistics, no calibration corpus.

**The fit is not seeded.** k-means draws an unseeded subsample, so this
artifact is reproducible in recipe and geometry but not bit-for-bit. That is
why two fits were scored rather than one.

## Verification

Every tensor was decoded **from the published artifact** and compared against
the bf16 source; no tensor exceeds 3x the artifact's own median
reconstruction error (median 0.0411). The bundled runtime was exercised as
the executing copy in a stock venv, not merely present in the folder.

Vision tower grafted in (333 tensors, 0.83 GiB); the text path is unchanged,
verified against the same KL cache.

### Multi-machine (exo) note

This artifact fits on one machine, but if you shard it across an
[exo](https://github.com/exo-explore/exo) cluster anyway, one guard is
required: VQ codebooks must **replicate rather than slice**. Stock exo tensor
parallelism slices them. The bundled `model.py` detects that and fails loudly
rather than generating fluent garbage, but it cannot fix the sharding itself.
Apply [exo PR #2268](https://github.com/exo-explore/exo/pull/2268) or use
[`noahzelezny/exo:vq-codebook-replicate`](https://github.com/noahzelezny/exo/tree/vq-codebook-replicate).

## Limitations

- The 35 GiB 8-bit is still measurably closer to bf16 (7.4 millinats against
  28.1). This is the closest VQ build here, not a lossless one.
- No perplexity, no throughput, and no task-suite scores for this artifact —
  see above.
- One asymmetry favours this build: the affine comparators quantize the MoE
  router, and this build leaves it at bf16. The bytes are trivial, but a
  router feeds an argmax over experts, so the effect is not bounded by the
  byte share. It is unmeasured.

## Provenance

Base model: Qwen/Qwen3.6-35B-A3B (Apache 2.0 — see the base model card for
license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX.
