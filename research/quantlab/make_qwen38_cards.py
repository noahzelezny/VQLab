#!/usr/bin/env python3
"""Generate the three Qwen3.8-27B cards from one template.

Written as a generator rather than three hand-edited files because the 35B
lineup drifted apart exactly that way: shared boilerplate that stopped being
shared. Any change to the common text lands in all three by construction.
"""
import io

# measured post-graft; comparators are OUR conversions - no community MLX
# quantization of this model exists
AFF = [("q2", 8.69, 1426.9, "46.1%", "16.435"), ("q3", 11.82, 187.8, "79.5%", "5.832"),
       ("q4", 14.95, 45.8, "89.8%", "5.206"), ("q6", 21.21, 3.71, "96.8%", "5.260"),
       ("q8", 27.48, 1.25, "98.5%", "5.241")]

BUILDS = [
 dict(repo="Qwen3.8-27B-VQ-3.9bpw", gib="12.47", res="11.61", geom="d=4, K=4096",
      bits="12-bit index into a per-tensor 4096-entry fp16 codebook", bpw="3.25",
      kl="85.8", top1="86.1%", ppl="5.229", ram="16 GB, comfortably",
      lead="**12.5 GiB — the smallest build here that we would actually recommend.**",
      claim="""Against our own 3-bit conversion at 11.8 GiB it more than halves the
divergence — 85.8 millinats against 187.8 — for 0.6 GiB more. Against the
4-bit at 15.0 GiB it gives up real quality to save 2.5 GiB. That is the trade
this rung exists for."""),
 dict(repo="Qwen3.8-27B-VQ-4.5bpw", gib="14.45", res="13.60", geom="d=2, K=256",
      bits="8-bit index into a per-tensor 256-entry fp16 codebook", bpw="4.25",
      kl="40.3", top1="90.1%", ppl="5.233", ram="16 GB, tightly",
      lead="**14.5 GiB — smaller than our 4-bit conversion, and closer to bf16.**",
      claim="""This is the cleanest comparison on the ladder: against the 4-bit affine
conversion it is **0.50 GiB smaller and 12% closer to bf16** (40.3 millinats
against 45.8), with 0.3 points better token agreement. Smaller and better on
the same instrument, no trade to weigh."""),
 dict(repo="Qwen3.8-27B-VQ-4.8bpw", gib="15.45", res="14.59", geom="d=2, K=512",
      bits="9-bit index into a per-tensor 512-entry fp16 codebook", bpw="4.75",
      kl="32.8", top1="90.8%", ppl="5.162", ram="24 GB, comfortably",
      lead="**15.5 GiB — the best quality here, and the only reproducible fit.**",
      claim="""Against the 4-bit affine conversion this build is **28% closer to bf16**
(32.8 millinats against 45.8) and 1.0 point better on token agreement, for
3.3% more bytes. That margin is 6.2x this family's measured fit-to-fit noise
floor.

**This is the one fit in the collection that is seeded.** k-means used a fixed
seed, so the artifact is reproducible from its recipe rather than only in
recipe. It also ran three times the Lloyd iterations of its siblings from
bit-identical starting centroids, which makes it the more converged fit —
every metric moved the right way. We are *not* claiming it is measurably
better than the shorter run: the improvement is below our resolution, not
shown to be zero, and a card should not turn that into a result."""),
]

TPL = """---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- qwen3.8
---

# TheDrainFlorist/{repo}

{lead}

A vector-quantized build of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple Silicon.
Stock `mlx-lm`, no patches — the VQ runtime ships inside the checkpoint as
`model.py`.

At the time of release no MLX-format quantization of this model had been
published, so the affine builds compared against below are our own
conversions rather than community artifacts — a weaker class of evidence
than a third party's, and worth knowing when reading the tables.

![where these releases sit](qwen38_ladder.png)

## Measured results

Scored against the bf16 teacher on the same corpus with an unmodified
`mlx-lm`. All sizes include the 333-tensor bf16 vision tower (0.858 GiB),
carried by every build here.

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | perplexity |
|---|---|---|---|---|
{table}

{claim}

**Rank these by KL, not perplexity.** On this instruction-tuned family
perplexity barely moves — the affine rungs above 3-bit span just 5.21 to
5.26, a 0.054 spread against a 0.0447 measurement floor — while divergence
from the teacher moves by a factor of 37 across the same range. Perplexity is an aggregate
over finite text and absorbs offsetting errors; KL measures distance to the
teacher's distribution directly.

## Runtime

**Not measured on this artifact.** No decode or prefill benchmark has been
run on this build, and quoting a sibling's figures would be a substitution
this project does not make. Resident memory is about {res} GiB — the disk
figure less the vision tower, which `mlx-lm` does not load.

Runs on a {ram} sized machine.

## Run it

```bash
pip install mlx-lm
python -m mlx_lm generate \\
  --model TheDrainFlorist/{repo} \\
  --prompt "Explain vector quantization briefly." \\
  --max-tokens 512
```

## How it was built

Vector quantization of the dense MLP trio at **{geom}**. Each {dim}-weight
subvector stores one {bits}. With an fp16 scale
per (row, 64 weights) that comes to {bpw} bits per weight over the quantized
surface; everything else in the model is 8-bit.

Every quantized tensor uses this one geometry: no depth schedule, no mixed
allocation.

**Codebooks are fit in pure weight space** — k-means over the weight
subvectors, no Hessian, no activation statistics, no calibration corpus.

{seedpara}

## Comparators

The affine rungs above are local conversions made with `mlx_lm.convert` at
its defaults, since no MLX build of this model has been published to compare
against. They are uniform quantizations at the bit width named, scored on the
same corpus and the same instrument as the VQ rungs.

## Where this stops paying

Above roughly 5 bits per weight the advantage reverses on this model: our
6-bit affine conversion reaches 3.7 millinats at 21.2 GiB, which no VQ rung we
measured approaches at that size. Builds larger than the ones released here
were measured and deliberately not published for that reason.

## Verification

Every tensor was decoded **from the published artifact** and compared against
the bf16 source; no tensor exceeds 3x the artifact's own median reconstruction
error. The bundled runtime was exercised as the executing copy in a stock
venv, not merely present in the folder. Vision tower grafted from the base
checkpoint and verified key-for-key against the official index, including the
channels-last patch-embedding layout that a naive rename gets silently wrong.


### Multi-machine (exo) note

This artifact fits on one machine, but if you shard it across an
[exo](https://github.com/exo-explore/exo) cluster anyway, one guard is
required: VQ codebooks must **replicate rather than slice**. Stock exo tensor
parallelism slices them. The bundled `model.py` detects that and fails loudly
with an explanatory error instead of silently generating fluent garbage that
reads as "a broken quant" — but it cannot fix the sharding itself. To actually
run tensor-parallel, apply [exo PR #2268](https://github.com/exo-explore/exo/pull/2268)
or run the ready branch
[`noahzelezny/exo:vq-codebook-replicate`](https://github.com/noahzelezny/exo/tree/vq-codebook-replicate).
Single-machine mlx-lm and pipeline sharding are unaffected.

## Paper

The method, the full three-model ladder, the negative results, and the
measurement rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

## Limitations

- Perplexity cannot rank builds on this family — see above.
- No throughput measurement, and no task-suite scores, for this artifact.
- The affine comparators are our own conversions, not community builds.
- Above ~5 bpw affine wins outright on this model; this collection stops
  below that line deliberately.
"""

SEED_PLAIN = """**The fit is not seeded.** k-means draws an unseeded subsample, so this
artifact is reproducible in recipe and geometry but not bit-for-bit. Margins
are therefore quoted against a measured fit-to-fit floor rather than against a
repeated build; on this family that floor is 2.085 millinats."""
SEED_SEEDED = """**The fit is seeded**, so this artifact is reproducible bit-for-bit from its
recipe and seed. Margins are still quoted against the family's measured
fit-to-fit floor of 2.085 millinats, which was measured from unseeded draws."""

for b in BUILDS:
    rows = []
    rows.append("| bf16 | 51.7 GiB | 0 | 100% | — |")
    for n, s, k, t, p in AFF:
        rows.append("| affine %s (ours) | %s GiB | %s | %s | %s |" % (n, s, k, t, p))
    rows.append("| **this model** | **%s GiB** | **%s** | **%s** | %s |"
                % (b["gib"], b["kl"], b["top1"], b["ppl"]))
    rows.sort(key=lambda r: float(r.split("|")[2].replace("*", "").replace("GiB", "").strip()))
    b2 = dict(b, table="\n".join(rows), dim=b["geom"].split("=")[1].split(",")[0],
              seedpara=SEED_SEEDED if "4.8bpw" in b["repo"] else SEED_PLAIN)
    out = "MODEL_CARD_QWEN38_%s.md" % b["repo"].split("-")[-1].replace(".", "_")
    io.open(out, "w", encoding="utf-8").write(TPL.format(**b2))
    print("wrote", out)
