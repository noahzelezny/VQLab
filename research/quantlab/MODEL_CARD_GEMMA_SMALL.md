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

**8.4 GiB text / 9.4 GiB with vision — a 26B MoE at the size of the 8.35 GiB e4b-8bit sidecar it was built to replace.**

**The design goal.** `gemma-4-e4b-it-8bit` (8.35 GiB) is a common sidecar
choice — small, fast, always resident. This build asks whether you can run
the *bigger* 26B-A4B model in that same slot instead. At full precision the
26B is the better literary model (84.62% vs e4b's 82.69% on generative,
position-debiased literary multiple-choice), so the question is whether
enough of that advantage survives compression to 8.4 GiB.

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
| decode | **~62 tok/s** |

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

## Limitations

- **Significantly below bf16 on blind literary judging (p=0.0016).** Chosen
  for size; the 18.7 GiB sibling is the quality build.
- Do not benchmark with perplexity.
- Audio is not included — this checkpoint ships zero audio tensors upstream.
