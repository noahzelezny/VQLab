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

# Qwen3.6-35B-A3B-VQ-3.8bpw

**15.7 GiB — smaller than the community 4-bit and a third closer to bf16.**

A vector-quantized build of
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) for Apple
Silicon. Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint as `model.py`.

## Measured results

Referee corpus, 2048-token windows, scored against the bf16 teacher on the
same files with an unmodified `mlx-lm`. All sizes include the bf16 vision
tower, which every build here carries:

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | perplexity |
|---|---|---|---|---|
| bf16 | 65.4 GiB | 0 | 100% | 4.7215 |
| mlx-community 8-bit | 35.1 GiB | 7.4 | 96.18% | 4.7150 |
| **this model** | **15.7 GiB** | **53.0** | **89.55%** | 4.7090 |
| mlx-community 4-bit | 19.0 GiB | 78.6 | 85.61% | 4.9154 |

**33% less divergence than the 4-bit, at 3.3 GiB smaller.** That is the
comparison this build exists for.

**Read the perplexity column carefully.** At 4.7090 this build sits slightly
*below* the bf16 teacher, and the 4-bit sits above it. That does not make it
better than bf16 — perplexity is an aggregate over finite text and absorbs
offsetting errors in both directions. KL measures distance to the teacher's
distribution directly and does not. Treat anything within about 1% of the
teacher as indistinguishable, and rank by KL.

Note the two metrics disagree in emphasis: KL and perplexity both favour this
build over the 4-bit, while top-1 agreement is 89.55% against the 8-bit's
96.18%. For a 256-expert MoE that is expected — discrete routing flips swap
one plausible token for another, which perplexity absorbs and argmax
agreement punishes. Both numbers are honest; quote both.

## Runtime

**Not measured on this artifact.** The two earlier releases in this family
carry decode figures; this one has not been benchmarked, and quoting a
sibling's throughput here would be a substitution this project does not make.
Peak memory will run close to the disk figure less the vision tower, which
`mlx-lm` does not load.

Comfortable on a 24 GB machine; workable on 16 GB at short context.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.8bpw \
  --prompt "Write a Python function that ..." \
  --max-tokens 512
```

## How it was built

Uniform vector quantization of the experts at **d=4, K=8192** — each
4-weight subvector stores one 13-bit index into a per-tensor 8192-entry fp16
codebook, with an fp16 scale per (row, 64 weights). Non-expert tensors are
8-bit. Experts are ~92% of the model, so that is where the bytes are.

There is no depth schedule and no mixed allocation: every expert tensor uses
the same geometry. Flat rungs are the reference points in this lineup because
no mixed-allocation build we measured beat the flat rung at its own size.

**Codebooks are fit in pure weight space** — k-means over the weight
subvectors, no Hessian, no activation statistics, no calibration corpus.

**The fit is not seeded.** k-means draws an unseeded subsample, so this
artifact is reproducible in recipe and geometry but not bit-for-bit. Margins
are therefore quoted against a measured fit-to-fit floor rather than against
a repeated build; on this family that floor is 0.214 mnats, so the 25-mnat
gap to the 4-bit is far outside it.

## Verification

Every tensor was decoded **from the published artifact** and compared against
the bf16 source; no tensor exceeds 3x the artifact's own median
reconstruction error. The packed artifact reproduced its unpacked twin's KL
score to every printed digit. The bundled runtime was exercised as the
executing copy in a stock venv, not merely present in the folder.

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

- Top-1 agreement is 89.55% against the 8-bit's 96.18%. If your workload is
  sensitive to exact token choice rather than distributional quality, the
  8-bit is measurably closer to bf16.
- No throughput measurement — see Runtime.
- No task-suite scores for this artifact.
- One asymmetry favours this build: the affine comparators quantize the MoE
  router, and this build leaves it at bf16. The bytes are trivial, but a
  router feeds an argmax over experts, so the effect is not bounded by the
  byte share. It is unmeasured.

## Provenance

Base model: Qwen/Qwen3.6-35B-A3B (Apache 2.0 — see the base model card for
license and usage terms). Quantization: TheDrainFlorist, 2026. Built with MLX.
