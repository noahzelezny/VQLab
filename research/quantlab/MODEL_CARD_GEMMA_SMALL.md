---
language:
- en
license: gemma
library_name: mlx
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26b-a4b-it
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- gemma-4
- multimodal
---

# gemma-4-26b-a4b-it-VQ-d4K256

**8.4 GiB text / 9.4 GiB with vision — matches the 15 GiB community 4-bit of the same model, at 63% of its size.**

**The design goal, and the honest outcome.** `gemma-4-e4b-it-8bit`
(8.35 GiB) is a common sidecar choice — small, fast, always resident. This
build asked whether the *bigger* 26B-A4B model could take that slot instead.

**Measured: it does not beat e4b-8bit on literary work.** Same instrument,
generative and position-debiased, n=104:

| model | size | literary MC |
|---|---|---|
| gemma-4-e4b-it-8bit | 8.35 GiB | **84.62%** |
| gemma-4-e4b-it bf16 | 19 GiB | 82.69% |
| **this model** | 9.43 GiB | **79.81%** |

If you are running e4b-8bit as a literary sidecar, this is not an upgrade —
keep e4b-8bit. What this build offers instead is a 26B-A4B **at 63% of the
size of the smallest community build of the same model** (15 GiB 4-bit,
which it matches at 79.81%), with vision included.

This is NOT the build to pick if you want bf16-quality prose — the 18.7 GiB
sibling in this collection exists for that and is a different goal
(bf16-indistinguishable, smallest size we could reach). This one is
size-first: the largest model that fits the sidecar slot.

Stock `mlx-lm`, vision included.

## Measured results

**Perplexity is NOT reported, deliberately.** Raw loglikelihood is invalid on
the gemma-4-it family — the RL-sharpened distribution collapses, so wikitext
ppl reads ~100-700 while the model writes fluent prose. Confirmed against HF
transformers on unquantized bf16, so it is a model property, not an MLX bug.

### Literary multiple-choice (generative, position-debiased)

| model | size | accuracy |
|---|---|---|
| 26b bf16 | 48 GiB | 84.62% |
| gemma-4-e4b bf16 | 19 GiB | 82.69% |
| mlx-community 4-bit | 15 GiB | 79.81% |
| **this model** | **9.4 GiB** | **79.81%** |

**Identical to the 15 GiB 4-bit at 63% of its size**, and it replaces a 19 GiB
bf16 sidecar at half the footprint.

### Blind literary judging vs bf16 — the honest part

60 anonymized pairs, key withheld, judged by claude-sonnet-5:

| | bf16 preferred | this model preferred | p |
|---|---|---|---|
| vs bf16 (48 GiB) | 34 | 12 | **0.0016** |

**bf16 is significantly better.** This build trades real quality for size.
The multiple-choice number above saturates and cannot see that gap; the
blind judging can. If you want quality indistinguishable from bf16, take the
18.7 GiB sibling in this collection, which draws 11-23 with 26 ties.

### KL divergence to bf16

| build | size | mean KL | top-1 agreement |
|---|---|---|---|
| 8-bit reference | 25 GiB | 441 | 79.95% |
| 18.7 GiB sibling | 18.7 GiB | 537 | 77.89% |
| **this model** | **9.4 GiB** | **3363** | **42.65%** |

80% is the practical ceiling on this metric, not 100% — a 128-expert top-8
MoE reroutes under any perturbation, so even 8-bit only reproduces bf16's
argmax ~80% of the time. Judge against that row.

## Runtime

Single M3 Ultra, macOS, stock `mlx-lm`, GPU otherwise idle:

| | |
|---|---|
| load time | ~5 s |
| peak memory | 8.4 GiB |
| decode | **~70 tok/s** |

*Measured on an M4 Max (128 GB), mlx-lm, 120-token greedy generation.*

The fastest build in this collection, and small enough to run beside a
larger model on a 32 GB machine.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model TheDrainFlorist/gemma-4-26b-a4b-it-VQ-d4K256 \
  --prompt "Summarize this in two sentences: ..." \
  --max-tokens 300
```

Vision tower included (356 tensors); the text path is bit-identical to the
text-only build.

## How it was built

Experts vector-quantized at **d=4, K=256** (2.25 bits/weight, byte-aligned so
no packing is needed); everything else at 8-bit.

## Verification

Every tensor decoded from the published artifact and compared against the
bf16 source; nothing exceeds 3x the artifact's own median reconstruction
error.


### Multi-machine (exo) note

These artifacts fit on one machine, but if you shard them across an
[exo](https://github.com/exo-explore/exo) cluster anyway, one guard is
required: VQ codebooks must **replicate rather than slice**. Stock exo
tensor parallelism slices them. This artifact's bundled `model.py` detects
that and fails loudly with an explanatory error (instead of silently
generating fluent garbage that reads as "a broken quant") — but it cannot
fix the sharding itself. To actually run tensor-parallel, apply
[exo PR #2268](https://github.com/exo-explore/exo/pull/2268) or run the
ready branch
[`noahzelezny/exo:vq-codebook-replicate`](https://github.com/noahzelezny/exo/tree/vq-codebook-replicate).
Single-machine mlx-lm and pipeline sharding are unaffected.

## Limitations

- **Significantly below bf16 on blind literary judging (p=0.0016).** Chosen
  for size; the 18.7 GiB sibling is the quality build.
- Do not benchmark with perplexity.
- Audio is not included — this checkpoint ships zero audio tensors upstream.
