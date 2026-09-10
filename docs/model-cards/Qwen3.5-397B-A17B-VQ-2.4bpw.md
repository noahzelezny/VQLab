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

**110.8 GiB — the daily driver, runs on a single 128 GB Mac.**

A vector-quantized build of [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
that fits and **generates on one 128 GB Apple Silicon machine** — no cluster,
no patches, stock `mlx-lm`.

## Changelog

### 2026-09-09 — runtime refresh

**Runtime refresh.** The bundled `model.py` is updated so downloaders run
exactly the code that was benchmarked; dense bundles now carry both runtimes.

- Faster prefill on affected geometries, measured per rung: a
  device-codebook kernel arm for large-codebook geometries (up to 1.46x on
  affected rungs), a ragged-subvector relaxation (up to 1.34x on affected
  rungs), a fused d8 arm, and a routing memo (≈2%). No blanket speedup is
  claimed across the lineup — gains apply only where the geometry engages
  the new paths.
- Output quality is unchanged: the kernel changes are bit-identical or
  1-ULP-equivalent, and the routing memo is bit-identical (logits checksum
  verified).
- Speculative decoding (MTP): on repos that ship
  `mtp-head-q6.safetensors`, the sidecar works with the exo fork branch
  `mtp-stage1` (github.com/noahzelezny/exo) — launch each node with
  `exo --mtp` (or set `EXO_MTP=1`). Note: with MTP enabled, exo serves
  requests sequentially (the batch engine has no MTP path), so leave it
  off for concurrent / multi-agent workloads.

### 2026-09 — bundle refresh (runtime update landed)

**Bundle refresh (runtime update landed).**

`model.py` updated: refreshed
VQ expert kernels (verified equivalent), a default buffer-cache ceiling
(`VQLAB_CACHE_LIMIT_GB` overrides, `0` disables) so transient prefill
allocations return to the OS as they free — long-prompt peak stays near
resident instead of a multiple of it — and dual-runtime loading. Weights
unchanged; prior revision pinnable. This rung exceeds the single-box gate
bar, so the update shipped through the release gate's **cluster smoke**: a
real generation on a 2-node exo pipeline, with the peer rank's copy
identity-checked before the generation counted.

## Measured results

All numbers measured on this exact artifact (not projected from a proxy),
reproduced bit-identically ×2, scored with an unmodified `mlx-lm` install.

| | this model (110.8 GiB) | spicyneuron 2.6bit (120.6 GiB) |
|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.7655** | 3.1843 |
| code perplexity (mixed-language) | **2.6383** | 2.6667 |

Runtime, single M4 Max 128 GB (macOS, stock `mlx-lm`):

| | |
|---|---|
| load time | ≈60 s |
| resident memory | 110.8 GiB (peak 117.7 GiB at 30k context) |
| context verified | **30,031 tokens**, zero swap growth |
| decode | **≈19–22 tok/s**, flat from 512 → 14k context |
| prefill | ≈40–50 tok/s (chunked, as mlx-lm does natively) |

Perplexities are corpus-specific: never compare them across different
corpora or eval harnesses, only against other models scored on the same
files. The wikitext margin (13.2%) is much larger than the code margin
(1.07%) — that asymmetry is real, so judge by your workload.

## Run it

```bash
pip install 'mlx-lm>=0.31.3'
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
- Close memory-heavy apps first; the model wants ≈111 GiB resident and
  peaks ≈118 GiB at long context.
- `SCOUT_VQ_DECODE_CHUNK` (env var) trades prefill speed for peak memory
  during long-prompt processing. The default auto-sizes from free memory;
  lower it (e.g. `16`) if you run close to the ceiling.
- Machines with more memory need none of this.

## Speculative decoding (MTP)

`mtp-head-q6.safetensors` (5.4 GiB) is the model's own MTP draft head —
the same file validated on the 2.2bpw rung (acceptance 0.72, single-box
via `vqlab serve --sidecar`). **Cluster speculative decoding is live** on the [`mtp-stage1`](https://github.com/noahzelezny/exo/tree/mtp-stage1)
branch of our exo fork (`EXO_MTP=1` on every node; outputs exactly the
base model's via rejection sampling). Measured on this family's 2.6bpw
rung on a 2-node pipeline: acceptance **0.85**, throughput at **parity**
with exo's stock decode — the head drafts well, but this family's stock
pipeline decode does not degrade with generation length, so there is
little for speculation to recover. Enable it to experiment; expect
parity; per-rung numbers for this artifact are not yet measured.

## Vision

The artifact includes the full 333-tensor vision tower at source precision
(0.85 GiB). `mlx-lm` is text-only for this architecture and ignores it;
[exo](https://github.com/exo-explore/exo) loads it from this folder
directly. `mlx-vlm` support requires its `model_file` loader hook
([PR #1926](https://github.com/Blaizzy/mlx-vlm/pull/1926), under review).

## Siblings

This is the middle of a three-size family, all from the same skeleton and
recipe, all measured the same way:

| | size | wikitext | code | needs |
|---|---|---|---|---|
| `VQ-2.2bpw` (accessibility) | 100.1 GiB | 3.1706 | 2.6988 | 128 GB Mac, roomy |
| **`VQ-2.4bpw` (this build)** | **110.8 GiB** | **2.7655** | **2.6383** | 128 GB Mac, tight |
| `VQ-3.1bpw` (quality) | 142.8 GiB | 2.3519 | 2.5987 | ≥192 GB or cluster |

## Methodology

**Mixed precision by layer sensitivity.** Not all weights deserve the same
bits. Attention, MoE routers, embeddings, and the output head stay at higher
precision — they are a small fraction of the parameters but errors there
propagate through every token. The MoE *experts* are ≈85% of the model and
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
(prefix-8192) and a mixed-language code corpus — every number reproduced
bit-identically twice, scored with an unmodified `mlx-lm`. Two corpora
because this family shows real domain asymmetry: larger codebooks buy far
more on prose than on code, so a single-corpus number would misrepresent the
trade. Task-suite evals (HellaSwag/PIQA/WinoGrande and friends) have **not**
been run; only what is reported above is measured.

## Verification

Release gates passed on this artifact before upload: file, index and
tokenizer checks, a verbatim match between the bundled runtime and its
source, and a generation smoke through the shipping runtime on Apple
Silicon. The upload path runs the gate itself and refuses to publish
without it.
## Limitations

- **Tight on 128 GB.** ≈118 GiB peak against ≈120 GiB usable leaves little
  room for other software. It runs; it is not roomy.
- This is a *thinking* model (Qwen3.5 family): by default it spends tokens
  reasoning before answering. Budget `max_tokens` accordingly.
- Distributed (exo) tensor-parallel serving needs one line in exo's own
  sharding rule — VQ codebooks must be replicated, not sliced. That change
  is not yet upstreamed; the one-line patch is in the experiment log and can
  be applied locally. `mlx-lm` itself is stock in that setup too: verified serving
  this model across two Macs with an unpatched `mlx-lm`, producing output
  identical to the patched run. Single-box users are unaffected.

## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

## Provenance

Base model: [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) — **Apache-2.0**.
This is a quantized derivative and inherits that licence; using it
means accepting the base model's terms.
Quantization: TheDrainFlorist, 2026.

Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab).
