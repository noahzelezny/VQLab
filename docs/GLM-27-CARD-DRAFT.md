# Launch model card draft — TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw

For Noah's review; nothing published, artifact dir untouched. Every number
traces to `research/quantlab/research/glm53-flash/LEDGER.md` / `TABLE.md` or
a direct read of the artifact. **Third pass (2026-09-03): rewritten into the
house card voice** — process narration, instrument-offset discussion and the
1.43x retraction story moved out of the card (they live in the LEDGER);
the card states the measured facts once, the way the shipped Flash/397B
cards do.

Outstanding before push (tracked here, not on the card):

- [ ] Full gate incl. smoke on this artifact (M4 re-gate step) — also
      captures the verbatim stock `mlx-vlm` invocation; the command below is
      the expected form, confirm against the smoke before push. The
      "Release gates passed" sentence under Provenance is written for the
      post-gate card and is NOT yet true — the push flow makes it true
      before upload, and publish refuses without the gate anyway.
- [ ] `check-bundle` against a pinned clean `vq_switch.py` commit.
- [ ] Confirm sidecar-outside-`vqlab serve` behavior (exo path) — the card
      currently says the shim ships with `vqlab serve` and stays silent on
      exo; verify that's the honest minimum.
- [ ] d4/K512 literary comparator cell (1.6166) under investigation; not
      load-bearing for any claim on this card, but if it moves, update the
      table row.

---

```yaml
---
language:
- en
- zh
license: mit
library_name: mlx
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- glm
---
```

# TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw

**101.9 GiB — GLM-5.3-Flash on one 128 GB Mac.**

A data-free vector-quantized build of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (598.5 GiB
bf16, vision) for Apple Silicon. Stock `mlx-vlm`, no patches — the VQ
runtime ships inside the checkpoint as `model.py`. Built with
[VQLab](https://github.com/noahzelezny/VQLab).

MoE experts at d=4/K=512 (9-bit packed codes), with eight expert layers
promoted to d=4/K=2048 (11-bit) — layers 20, 27, 29, 31, 33, 34, 35, 39,
chosen by measured single-layer effect. Attention, embeddings and the output
head stay 8-bit affine; norms, routers and the full 347-tensor vision tower
stay bf16.

The affine builds compared against below are our own conversions of the same
base, made with the same tooling, scored on the same instrument.

## What vector quantization is

Affine quantization (`q3`/`q4`/`q6`) rounds each weight independently onto a
uniform grid; below about 5 bits the grid is too coarse for the weight
distribution. VQ stores shapes instead of numbers: weights are grouped into
vectors of `d` consecutive values, each stored as an index into a codebook
of `K` representative vectors learned by k-means over the weight matrix
itself — `log2(K)/d` bits per weight, with the codebook free to put entries
where the weights actually are. The fit is data-free: no calibration corpus,
no activations, no teacher forward.

This build goes beyond our published recipe. The paper
([*Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits*](https://doi.org/10.5281/zenodo.22136000), CC BY 4.0)
establishes the VQ-over-affine result on flat rungs — one geometry applied
uniformly. This model uses the newer mixed-codebook strategy: individual
expert layers promoted to a richer codebook by measured effect, every
candidate promoted one at a time and scored on the assembled model. That
buys 1.93x the KL-per-GiB of adding bits uniformly (details under **How it
was built**).

## Requirements

The `glm5_next` architecture ships in **released `mlx-vlm` 0.6.17** — a
normal `pip install`, not a fork or an unmerged PR. The VQ runtime needs no
patches: it ships as `model.py` inside the checkpoint, declared via
`model_file` in `config.json`, and resolves under both `mlx-lm` and
`mlx_vlm`.

```bash
pip install mlx-vlm

python -m mlx_vlm generate \
  --model TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 512
```

exo-ready: `config.json` carries `vision_config` and `image_token_id`, and
the vision tower ships bf16. For cluster serving use the
[`vq-serving`](https://github.com/noahzelezny/exo/tree/vq-serving) branch.

## Measured results

Referee: 2048 tokens; prose = WikiText, code = public mlx corpus (pinned
manifest), literary = Gutenberg. KL is against the bf16 teacher's cached
top-64 logits (captured mass 0.9906 on every row). This model's row was
measured on the published artifact itself, all three corpora.

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | prose ppl | code ppl | literary ppl |
|---|---|---|---|---|---|---|
| VQ d4/K512 (this build's base) | 98.5 GiB | 348.82 | 84.0% | 2.5743 | 1.7107 | 1.6166 |
| **this model** | **101.9 GiB** | **291.46** | **86.2%** | **2.3978** | **1.6696** | **1.4843** |
| VQ d4/K2048 (uniform) | 116.3 GiB | 199.53 | 88.6% | 2.1954 | 1.6187 | 1.3402 |
| affine q3 (ours) | 129 GiB | 377.08 | 83.1% | 2.6824 | 1.7842 | 1.4731 |
| VQ d4/K8192 (uniform) | 134.0 GiB | 94.54 | 92.1% | 2.0379 | 1.5475 | 1.2154 |
| affine q4 (ours) | 166 GiB | 98.34 | 91.9% | 2.0263 | 1.5718 | 1.2025 |
| affine q6 (ours) | 239 GiB | 13.47 | 97.1% | 1.9285 | 1.4929 | 1.1660 |
| bf16 teacher | 598.5 GiB | 0 | 100% | 1.9024 | 1.4888 | 1.1580 |

**This is the fits-a-128 GB-Mac rung.** Against affine q3 it is 27 GiB
smaller *and* ~23% better on KL (291.46 vs 377.08), with clear wins on prose
and code; q3 edges it narrowly on literary (1.4731 vs 1.4843), the corpus
this family memorized hardest, and the gap closes with bits — the 116 GiB
rung wins literary outright. It is not q4-class: reaching q4 quality from
this family costs 134 GiB (the d4/K8192 rung, which matches or beats affine
q4 on every axis at 32 GiB less), and affine q4 itself is a 166 GiB
download. At this rung's memory budget it is the best thing we have
measured.

**Rank these by KL, not perplexity.** Perplexity is an aggregate over finite
text and absorbs offsetting errors; KL measures distance to the teacher's
distribution directly. KL here is prose-only (the teacher cache is a prose
cache).

**Do not compare these perplexities to any other model family.** The bf16
teacher has near-verbatim memorized the public corpora used here (mean top-1
probability 0.857 on prose, measured, with a causality test ruling out a
leaky mask), so absolute perplexity is contamination-dominated. KL to that
teacher stays fully valid — a sharp teacher is *harder* to track.

## Runtime

Measured on a Mac Studio M4 128 GB, single box, warm (cold first cycles page
badly at ~101 GiB resident; discard one before benchmarking):

| | |
|---|---|
| decode, stock generate | **19.7 tok/s** |
| decode, MTP sidecar (`vqlab mtp-generate`) | 19.99 tok/s, acceptance 0.827 |
| prefill | ~88 tok/s (lower bound, derived from wall time) |

## Speculative decoding (MTP) — optional sidecar

This repo includes `mtp-head-q6.safetensors` (6.09 GiB): GLM's own
multi-token-prediction head — `layers.45`, 889 tensors, 13.84 GiB as a bf16
graft, packed to q6. It is never named in the weight index, so stock loaders
ignore it entirely; it costs nothing unless you opt in by name.

**When enabled it adds ~6.3 GiB resident** (head weights + its cache).
Acceptance is **0.827**, and the trunk verifies every drafted token by exact
rejection sampling, so the output distribution is exactly the base model's.
**On a single box it runs at parity with plain decode** (19.99 vs
19.7 tok/s) — download it for the draft quality and the serving setups where
speculation pays: pipelined multi-node decoding (in validation) and batch
serving. No single-box speedup is claimed.

The sidecar path includes an absorbed-MLA fix worth knowing about: GLM's
sparse attention takes its absorbed route only at L=1 upstream, so a 2-token
verify paid a full unabsorbed expansion of the latent cache — a measured 23x
per-layer tax that grows with context. The bundled runtime takes the
absorbed route for all L ≤ 8, byte-for-byte identical topk/sparse-mask
construction, verified numerically equivalent (max abs difference one bf16
ULP, indexer cache bitwise identical). `vqlab serve` installs it; without
it, the sidecar is a net loss.

To use it:

```bash
git clone https://github.com/noahzelezny/VQLab && cd VQLab
python3 -m venv .venv && source .venv/bin/activate
pip install .
python -m vqlab.cli serve \
  --model TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw \
  --sidecar mtp-head-q6.safetensors
```

OpenAI-compatible API on localhost; `vqlab mtp-generate` for one-shot CLI
use. Without `--sidecar`, nothing about the model changes.

The `mtp-head-q6.safetensors` files in our lineups (Qwen Flash 2.14 GiB,
Qwen 397B 5.41 GiB, GLM 6.09 GiB) are different heads with different
geometry and share only a filename — never cross-copy them between families.

## Memory, measured externally

Process RSS sampled at 5 Hz from outside the process, M4 128 GB, single box:

- **61.1 GiB** peak for load + 2048-token prefill + 128-token decode.
  Weights are memory-mapped, so RSS counts only pages actually touched —
  useful for comparing artifacts on the same instrument, **not for capacity
  planning**.
- **~100.9 GiB** sustained resident once routing has touched the experts;
  budget this. Trunk + MTP head is ~107 GiB — it runs on a 128 GB box and is
  tight there; close memory-heavy applications.

Long-prompt peaks stay near resident: the bundled runtime caps MLX's
buffer-reuse cache by default (`VQLAB_CACHE_LIMIT_GB` overrides, `0`
disables), measured free on wall time. Under exo, set
`EXO_MLX_CACHE_LIMIT_GB=6` and `EXO_MLX_MEM_LIMIT_GB` a few GiB under
physical RAM so an overrun is a traceback rather than a frozen Mac; a
26,423-token prompt through a 2-node pipeline held per-chunk transients
under 2 GiB, flat across all 13 chunks. If you run your own serving loop,
chunk the prefill (2048) and call `mx.eval([c.state for c in cache])` plus
`mx.clear_cache()` after every chunk — the chunk size only bounds the peak
if each chunk is actually forced.

## How it was built

Fitted data-free from the bf16 checkpoint — k-means / Lloyd over weight
subvectors, seed 1234, no Hessian, no activation statistics, no calibration
corpus. The base is a uniform d=4/K=512 fit of the 42 expert layers (L3–L44;
L45 is GLM's MTP layer and is never quantized), 98.55 GiB at 2.635 bpw.
Eight expert layers are then promoted to d=4/K=2048 (+0.42 GiB each), giving
the 101.93 GiB shipped size. Codes are packed sub-byte into uint32 words
(9 bits at K=512, 11 at K=2048) with an fp16 scale per (row, 64 weights).
The "2.7bpw" in the name is derived from the measured sizes; GiB is the
measured quantity and is what the card quotes.

**The eight layers were chosen by measurement, not by a heuristic**, and the
comparison was run at identical bytes and geometry:

| 8-layer build @ 101.93 GiB | KL | prose ppl | top-1 |
|---|---|---|---|
| picked by a layer-leverage probe | 333.59 | 2.5461 | 84.6% |
| bottom-8 by the same probe (control) | 331.93 | 2.4948 | 84.8% |
| **best-8 by measured single-layer effect (shipped)** | **293.84** | **2.4014** | **85.7%** |

The control beat the probe's pick — the leverage probe carries no usable
information about output damage for this family. What works is direct
measurement: all 42 layers promoted one at a time and scored (26 help, 16
hurt; the surface is non-monotonic). Targeting this way is 1.93x the
efficiency of buying bits uniformly, capturing 37% of the full K512→K2048
step's gain for 19% of its bytes. Layer effects are base-specific — the same
sweep on a different rung produced wrong-sign predictions — so this table
does not transfer to other rungs or families. The seed-noise floor for this
geometry is 6.32 mnats on KL, below every margin argued from here.

## Known limitations

- **This is the aggressive end of the ladder.** Prose perplexity is 1.26x
  the teacher's. If you have the memory, the 134 GiB rung reaches q4-class
  quality (not yet published).
- **q3 edges it on literary** (1.4731 vs 1.4843) — the one axis where
  smaller-and-better does not hold. VQ damage lands hardest on the most
  memorized corpus, so test a literary-heavy workload yourself.
- **Tight on 128 GB with the sidecar** (~107 GiB resident). Cold runs page;
  discard a cycle before benchmarking.
- **MTP is single-box parity, not a speedup** (19.99 vs 19.7 tok/s), and it
  requires the bundled absorbed-MLA fix — without it the sidecar is a net
  loss.
- **Decode small-M kernel behaviour** is a known open frontier for this
  lineup, not a property of the quantization quality.

## Provenance and gates

Base model:
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) —
**MIT licensed**; this is a quantized derivative and inherits that license.
Quantization: TheDrainFlorist, 2026. Release gates passed on this artifact:
file/index/tokenizer checks, bundle-runtime verbatim match, and a generation
smoke through the shipping runtime on Apple Silicon; the full three-corpus
referee row was measured against the published artifact directory itself.

The upstream authors ask that their technical report be cited in research
use:

```bibtex
@misc{glm5team2026glm5vibecodingagentic,
  title={GLM-5: from Vibe Coding to Agentic Engineering},
  author={GLM-5-Team and others},
  year={2026},
  eprint={2602.15763},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

Quantization method and measurement rules:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22136000) (CC BY 4.0).

Local artifact: `glm53_vq_packed_mix_best8`.
