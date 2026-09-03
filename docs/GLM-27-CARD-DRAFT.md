# Launch model card draft — TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw

For Noah's review. **Nothing here is published**; the artifact directory was
not touched. Every number below traces to
`research/quantlab/research/glm53-flash/LEDGER.md`, its `TABLE.md`, or a
direct read of the artifact's `config.json` / file sizes. Markers
`[TO MEASURE: …]` and `[VERIFY: …]` are where the template wants a number the
ledger does not have — they are deliberate, not placeholders to be guessed at
before push.

**Second pass, 2026-09-03.** Closed since the first draft: the full
three-corpus referee row for the published artifact (measured, with its
instrument offset stated); the license (read from the upstream card — MIT);
the external-RSS peak; the decode and prefill figures. Rewritten: the MTP
section, which claimed a 1.43x speedup that was loop-relative and is now
stated as single-box parity. Added: the standard VQ explainer and an explicit
statement that this build's mixed-codebook strategy goes beyond the published
paper's single-geometry recipe. Three `[VERIFY: …]` markers remain, all about
what has actually been *executed* on this artifact rather than about numbers:
the sidecar-outside-`vqlab serve` path, a pasted stock-`mlx-vlm` invocation,
and the release gate.

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

> **License resolved (2026-09-03).** Read directly from the upstream card at
> `huggingface.co/zai-org/GLM-5.3-Flash` (`raw/main/README.md`): the base model
> is **MIT**, `language: [en, zh]`, `pipeline_tag: image-text-to-text`. The
> frontmatter above mirrors it; `library_name` is `mlx` because this is an MLX
> conversion, not a `transformers` checkpoint. No separate `license_link` is
> declared upstream, so none is claimed here — MIT is a named SPDX identifier
> and needs no local `LICENSE` copy to resolve. `pipeline_tag` matches upstream
> *and* the artifact: this checkpoint carries a live vision tower and
> `image_token_id`.
>
> Upstream asks for a citation of their technical report when the model is used
> in research; it is reproduced under Provenance.

---

# TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw

**101.9 GiB — GLM-5.3-Flash on one 128 GB Mac.**

A data-free vector-quantized build of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) for Apple
Silicon, fitted from the bf16 checkpoint (598.5 GiB) with no calibration
corpus. Built with [VQLab](https://github.com/noahzelezny/VQLab).

MoE experts at d=4/K=512 (9-bit packed codes), with **eight expert layers
promoted to d=4/K=2048** (11-bit codes) — layers 20, 27, 29, 31, 33, 34, 35,
39, chosen by measured single-layer KL effect, not by a heuristic. Attention,
embeddings and the output head stay at 8-bit affine; norms, routers and the
full 347-tensor vision tower stay bf16.

The affine builds compared against below are our own conversions of the same
base, made with the same tooling and scored on the same instrument.

## What vector quantization is, and why it wins down here

Affine quantization — the `q3` / `q4` / `q6` builds everyone ships — rounds
**each weight independently** onto a uniform grid, storing a scale and a zero
point per group of weights. It is simple and it is very good at 8 bits. Its
problem is arithmetic: at 3 bits a weight has 8 possible values, at 2 bits it
has 4, and no amount of tuning the scale changes the fact that the grid is
uniform and one-dimensional. Below about 5 bits the grid is simply too coarse
for the distribution it is being asked to represent.

Vector quantization stores **shapes instead of numbers**. Weights are grouped
into vectors of `d` consecutive values, and each vector is stored as an index
into a *codebook* — a table of `K` representative `d`-dimensional vectors
learned by k-means over the weight matrix itself. The stored cost is
`log2(K)/d` bits per weight. The codebook is free to put its entries wherever
the weights actually are, in `d` dimensions, rather than on an evenly spaced
line: at d=4/K=512 a single index selects one of 512 learned 4-D shapes for
2.25 bits of code. That is why codebooks beat affine scales at low bpw — the
representation adapts to the weight distribution instead of tiling it.

The fit is **data-free**: k-means over the weights, no calibration corpus, no
activations, no teacher forward. Nothing about your data enters the model.

**This build goes beyond the published recipe.** Our paper,
[*Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits*](https://doi.org/10.5281/zenodo.22136000) (CC BY 4.0), establishes
the VQ-over-affine result on **flat rungs** — one `(d, K)` geometry applied
uniformly to every quantized layer — and explicitly concludes there that mixed
allocation is a *size-targeting* tool rather than a quality one, because no
mixed build in that work beat the flat rung at or above its own size. It also
reports that per-layer sensitivity *probes* rank layers in ways that do not
survive contact with assembled-model scores on MoE.

This model uses the **newer mixed-codebook strategy** that those two negatives
motivated: instead of one geometry everywhere, individual expert layers are
**promoted to a richer codebook by measured effect** — every candidate layer
promoted one at a time and scored on the assembled model, with the promotions
chosen from those measurements rather than from a proxy. That turns mixed
allocation into a quality tool, at 1.93x the KL-per-GiB efficiency of buying
bits uniformly. The paper is the prior work this extends; the method, its
failure modes and its limits are documented in **How it was built** below, and
the probe that the paper distrusted is shown failing again here.

## Measured results

Referee: 2048 tokens; prose = WikiText, code = public mlx corpus (pinned
manifest), literary = Gutenberg. KL is against the bf16 teacher's cached
top-64 logits (`glm53_teacher_topk_prose`, captured mass 0.9906 on every row).

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | prose ppl | code ppl | literary ppl |
|---|---|---|---|---|---|---|
| VQ d8/K16384 (ours, dead rung) | 80.9 GiB | 692.25 | 74.8% | 3.6339 | 1.9619 | 2.9562 |
| VQ d4/K512 (this build's base) | 98.5 GiB | 348.82 | 84.0% | 2.5743 | 1.7107 | 1.6166 |
| **this model** | **101.9 GiB** | **291.46** | **86.2%** | **2.3978** | **1.6696** | **1.4843** |
| VQ d4/K2048 (uniform) | 116.3 GiB | 199.53 | 88.6% | 2.1954 | 1.6187 | 1.3402 |
| affine q3 (ours) | 129 GiB | 377.08 | 83.1% | 2.6824 | 1.7842 | 1.4731 |
| VQ d4/K8192 (uniform) | 134.0 GiB | 94.54 | 92.1% | 2.0379 | 1.5475 | 1.2154 |
| affine q4 (ours) | 166 GiB | 98.34 | 91.9% | 2.0263 | 1.5718 | 1.2025 |
| affine q6 (ours) | 239 GiB | 13.47 | 97.1% | 1.9285 | 1.4929 | 1.1660 |
| bf16 teacher | 598.5 GiB | 0 | 100% | 1.9024 | 1.4888 | 1.1580 |

> **Measured 2026-09-03 — this row is now a full three-corpus score.** All six
> numbers for this model were measured in one session on one instrument
> (streamed referee, 2048 tokens, `--kl-cache glm53_teacher_topk_prose`),
> against the **published artifact itself**, not against the local build. Two
> things came out of that run worth stating on the card:
>
> - **The shipped repack is value-neutral.** The published artifact and the
>   local build it was packed from score *bit-identical* on prose
>   (2.397798 / 291.4628 / 0.8618). The re-bundle changed the kernel and not
>   one output bit.
> - **This row carries a ~0.1–0.2% favourable instrument offset** against the
>   comparator rows, which were measured earlier. Re-measuring comparators
>   today reproduced them to −0.08% (K512 prose), −1.65 mnats (K512 KL) and
>   −0.12% to −0.18% (code) — consistently small and same-signed. That is
>   above our ~0.04% cross-instrument floor and far below every margin this
>   card argues from; no comparison here changes order or sign. It is stated
>   rather than buried.
>
> One published comparator cell — d4/K512 *literary*, 1.6166 — did not
> reproduce today (1.6285). It is flagged as open in the ladder and is not
> load-bearing for anything on this card.

**What this rung is for: it is the fits-a-128 GB-class-Mac point.** Said
plainly, so nobody has to infer it:

- **Against q3, it wins on both axes at once.** 27 GiB smaller than our affine
  q3 (101.9 vs 129 GiB) *and* ~23% less KL damage (291.46 vs 377.08), with
  +3.1 pt top-1 agreement. Smaller *and* better is the whole claim, and this
  rung supports it.
  **With one measured exception, stated rather than hidden:** on the literary
  corpus q3 is a hair ahead (1.4731 vs our 1.4843). We win prose (2.3978 vs
  2.6824) and code (1.6696 vs 1.7842) clearly and lose literary narrowly. This
  is the memorization deficit this family shows everywhere: literary is the
  most near-verbatim-memorized corpus, and it is exactly where VQ's damage
  lands hardest and affine's lands lightest. It closes with bits — the 116 GiB
  rung already wins literary outright (1.3406).
- **It is not q4-class, and q4-class is a different machine.** Affine q4 is
  166 GiB — that is a 256 GB box, not a 128 GB one. Reaching q4-class quality
  from *this* family costs 134 GiB (our uniform d4/K8192 rung, which matches or
  beats q4 on every axis at 81% of its size) and that wants 192 GB. Neither is
  a 128 GB download. **The 134 GiB rung is not yet published.**

So the comparison this model should be judged on is against q3 and against
nothing above it: at the memory budget where it lives, it is the best thing we
have measured. If you have the RAM for q4-class, you want a different artifact
and should wait for the 134 GiB rung rather than expect this one to close the
gap.

**Rank these by KL, not perplexity.** Perplexity is an aggregate over finite
text and absorbs offsetting errors; KL measures distance to the teacher's
distribution directly.

**Contamination note — do not compare these perplexities to any other model
family.** The bf16 teacher has near-verbatim memorized the public corpora
used here (mean top-1 probability 0.857 on prose; 68% of positions above
0.9 — measured, with a direct-forward causality test ruling out a leaky
mask). Absolute perplexity for this family is contamination-dominated. KL to
that teacher stays fully valid and is *stricter* here, because a sharp
teacher is harder to track. That is why KL is the ranking column.

**Caveat on the two prose-only columns.** Our "KL" is prose-only — the
teacher cache is a prose cache and `--kl-cache` is passed on the prose run
only (LEDGER 2026-09-01, "METRIC CORRECTION"). Ranking by KL weights prose at
100%. Where full three-corpus scores exist, summed added nats is the better
selection metric.

## Runtime

Measured on a **Mac Studio M4, 128 GB**, single box, warm:

| | tok/s |
|---|---|
| decode, plain `mlx-lm generate` | **19.7** |
| decode, `vqlab mtp-generate` (sidecar, acceptance 0.827) | **19.99** |

**Prefill: ~88 tok/s (derived, not directly instrumented).** A single-box
load + 2013-token prefill + 128-token decode run took **29.4 s** wall for the
prefill+decode phase. Subtracting the decode at the measured 19.7 tok/s
(128 / 19.7 = 6.5 s) leaves ~22.9 s for 2013 prompt tokens, i.e. **~88 tok/s**.
That is a subtraction, not a measurement: it charges all non-decode wall time
to prefill and so is a *lower* bound on the true prefill rate. Treat it as an
order-of-magnitude figure until a dedicated prefill timer is run.

**Cold, uninterleaved numbers on this model are worthless** and we will not
quote any: an early session measured 0.56 → 2.56 tok/s across arms purely from
paging, at ~100.9 GiB resident on a 128 GiB box. Discard a full cycle before
you believe a number.

## Speculative decoding (MTP) — optional sidecar

This repo includes `mtp-head-q6.safetensors` (**6.09 GiB**): GLM's own
multi-token-prediction head — `layers.45`, 889 tensors, 13.84 GiB as a bf16
graft, packed to q6. It is never named in the weight index, so stock loaders
ignore it entirely; it costs nothing on disk-to-RAM unless you opt in.

**When enabled it adds ~6.3 GiB resident** (head weights plus its cache) on
top of the trunk. The trunk verifies every drafted token by exact rejection
sampling, so the output distribution is exactly the base model's.

### No single-box speedup is claimed today

Be clear about what the sidecar does and does not buy right now. Measured on
this exact artifact, same M4, single box:

| | tok/s |
|---|---|
| plain `mlx-lm generate` | 19.7 |
| `vqlab mtp-generate` (sidecar) | **19.99** |

That is **parity**, not a speedup. Acceptance is **0.827** — genuinely
excellent, and the part of the system that is working. The draft head is
predicting well; the throughput simply is not being converted into wall-clock
gain on a single box yet.

An earlier draft of this card quoted **1.43x** (4.60 → 6.6 tok/s). That figure
was **loop-relative** — both arms ran inside the VQLab serve loop, and the
4.60 tok/s baseline was the loop, not the model. Against plain `mlx-lm`
generation at 19.7 tok/s, the ratio disappears. The 1.43x is withdrawn as a
user-facing claim; the underlying A/B was real, it was just measured against
the wrong baseline.

**Where the payoff arrives.** Two places, both known, neither shipped:

- **Cluster pipeline decoding** — speculation pays most where the verify step
  is pipelined across nodes rather than serialized on one box. In validation.
- **Serve-loop optimization** — the 4.60 tok/s figure above is the size of the
  overhead sitting between the sidecar and its benefit. That is our bug, and
  closing it is what turns 0.827 acceptance into throughput.

Ship it because acceptance is excellent and it costs you nothing to have; do
not download it expecting a faster single box today.

### The absorbed-MLA shim

The sidecar is only at parity rather than well behind because of a runtime
fix, and this is worth knowing. GLM's sparse attention takes an *absorbed* MLA
route only
when `L == 1`; a 2-token verify fell off it and paid a full unabsorbed
expansion of the entire latent cache, which is independent of L and grows
without bound in context (measured per-layer at 13312 cached rows: 1.263 ms
at L=1 vs 29.152 ms at L=2, a 23x tax). Pre-fix, end-to-end MTP on this
artifact was **1.05x** — the head worked, the verify was too expensive. The
shim takes the absorbed route for all L ≤ 8, leaving the topk / sparse-mask
construction byte-for-byte as upstream wrote it. Numerical equivalence was
verified rather than asserted: bf16 at real dims, max abs difference
2.4e-04–4.9e-04 = exactly one bf16 ULP at that magnitude, with indexer cache
state bitwise identical.

Ceiling with perfect acceptance is 1.81x, and break-even is r < 1.81 — 0.827
acceptance was never the limiting factor, which is the whole point of the
section above: the head is not what is holding the speedup back.

To use it, serve with VQLab:

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

> Note for the reviewer: the sidecar path requires the absorbed-MLA shim
> (`src/vqlab/glm5_shim.py`, merged 3cf0af9). `vqlab serve` installs it.
> [VERIFY: whether a user running the sidecar outside `vqlab serve` — e.g.
> under exo — gets the shim, and say so explicitly in this section. Also
> confirm the head is bundled into the *published* runtime path, not only the
> working tree.]
>
> The three `mtp-head-q6.safetensors` files in this lineup (Flash 2.140 GiB,
> 397B 5.412 GiB, GLM 6.094 GiB) are **different heads with different
> geometry** and share only a filename. Never cross-copy them.

## Requirements

The `glm5_next` architecture ships in **released `mlx-vlm` 0.6.17** — a
normal `pip install`, not a fork or an unmerged PR. The VQ runtime itself
needs no patches: it ships inside the checkpoint as `model.py`, declared via
`model_file` in `config.json`, and the bundle resolves under both `mlx-lm`
and `mlx_vlm`.

```bash
pip install mlx-vlm
```

[VERIFY: the exact stock generate invocation for this artifact under
mlx-vlm 0.6.17, run once and pasted verbatim. The release-gate smoke is the
right place to capture it.]

exo-ready: `config.json` carries `vision_config` and `image_token_id`, and
the vision tower ships bf16. For cluster serving use the
[`vq-serving`](https://github.com/noahzelezny/exo/tree/vq-serving) branch
(codebook-replicate guard plus the per-chunk prefill eval and memory knobs
below).

## Memory

- **Trunk resident: ~100.9 GiB**, measured on the M4 during the A/B session.
  Download is 101.9 GiB (19 shards); the repo total including the sidecar is
  **108.0 GiB**.
- **Trunk + MTP head wants a 128 GB single box and is tight there** —
  ~107 GiB against ~120 GiB usable. It runs; it is not roomy. Close
  memory-heavy applications, and expect a cold first cycle to page.
- **Peak ≈ resident plus a bounded transient.** The long-prompt transient on
  this model was measured and then eliminated: intermediates freed inside a
  prefill chunk park in MLX's buffer-reuse cache, which is excluded from
  `get_active_memory` — which is why an early instrument reported ~2.5 G/chunk
  while the system showed +24 G (M3) / +43 G (M4). Capping the reuse cache
  fixes it at no measured cost.

**Long prompts.** On a 26,423-token prompt through a 2-node exo pipeline with
`EXO_MLX_CACHE_LIMIT_GB=6`: reuse cache 0.0–0.2 G on **every** chunk on
**both** ranks, per-chunk transient under 2 G and flat across all 13 chunks,
per-rank peaks **51.8 G (M3) / 66.0 G (M4)**, wall 134 s against 128–137 s
uncapped — i.e. free. Additionally set `EXO_MLX_MEM_LIMIT_GB` a few GiB under
physical RAM so an overrun is a traceback and not a frozen Mac (the M4 in
this lineup runs 82).

If you run your own serving loop rather than `vqlab serve` or exo: keep the
prefill chunked (2048) and call `mx.eval([c.state for c in cache])` plus
`mx.clear_cache()` after **every** chunk. Without the per-chunk eval, MLX's
lazy graph can accumulate the entire prompt before evaluating; the chunk size
only bounds the peak if each chunk is actually forced.

**External-RSS peak: 61.1 GiB** (M4, 128 GB, single box). Measured the way the
other cards in this lineup measure it — process RSS sampled at 5 Hz from
*outside* the process, across load + a 2013-token prefill + 128 decoded
tokens. This is the whole-model, single-box figure, not a per-rank one.

Read it with one caveat, because it is lower than the 101.9 GiB download and
that surprises people: weights are memory-mapped, so RSS counts only the pages
actually faulted in during a run, not the full file. **Do not plan capacity
from 61.1 GiB.** Plan from the ~100.9 GiB resident figure above — a longer
prompt, a larger batch or a second pass will touch more of the file. The RSS
number is useful for comparing this artifact against the others in the lineup
on the same instrument, and for nothing else.

*(Superseded: an earlier draft quoted 66.0 G from the 26k 2-node pipeline
session. That was a per-rank MLX-side figure carrying one rank's shard, and it
was never this model's peak memory.)*

## How it was built

Fitted **data-free** from the bf16 checkpoint — k-means / Lloyd over weight
subvectors, seed 1234, no Hessian, no activation statistics, no calibration
corpus. Local artifact: `glm53_vq_packed_mix_best8`.

The base is a uniform d=4/K=512 fit of the 42 expert layers (L3–L44; L45 is
GLM's MTP layer and was never quantized), packed to 98.55 GiB at 2.635 bpw
overall, 2.471 bpw in the expert region — within 1% of the projection from
`bpw = log2(K)/d + 0.25` and param counts. Eight expert layers are then
promoted to d=4/K=2048 (+0.42 GiB each, +3.38 GiB total), which is the
101.93 GiB shipped size.

**The eight layers were chosen by measurement, and the measurement matters
more than the result.** Three things were tried, at identical bytes and
identical geometry:

| 8-layer build @ 101.93 GiB | KL | Δ vs base | prose | top-1 |
|---|---|---|---|---|
| picked by leverage probe (`local_rel` ranking) | 333.59 | +15.23 | 2.5461 | 84.6% |
| bottom-8 by leverage (control) | 331.93 | +16.89 | 2.4948 | 84.8% |
| **best-8 by measured single-layer effect (shipped)** | **293.84** | **+54.98** | **2.4014** | **85.7%** |

*(These three arms are quoted at their original sweep values, all measured
against each other on one instrument in one session — which is what makes the
comparison valid. They are the pre-2026-09-03 instrument, hence 293.84 here
against 291.46 in the main table; the offset applies to all three arms equally
and changes no ordering.)*

The control **beat** the probe's pick. The layer-leverage probe carries no
usable information about output damage for this family, and three of its
eight "highest-leverage" layers were among the ones that *hurt* KL when
promoted alone. What works is promoting the layers whose single-layer effect
was directly measured: all 42 promoted one at a time and scored (26 help, 16
hurt — the surface is non-monotonic; L27 alone is worth +19.08 mnats for
0.42 GiB). Targeting this way is **1.93x the efficiency of simply buying
bits uniformly** (16.27 vs 8.42 mnats/GiB), capturing 37% of the full uniform
K512→K2048 step's gain for 19% of its bytes.

Two limits on that, both measured: greedy selection on an interacting
non-monotonic surface carries no optimality guarantee (a DP solve at the same
size reached KL 291.69, 2.15 better), and **layer effects are base-specific**
— the same sweep applied to a different rung produced wrong-sign predictions
on a single-layer test, so a per-rung sweep is required and this table does
not transfer.

The seed-noise floor for this geometry is **6.32 mnats** on KL (an unseeded
refit of the same fit). Splice-vs-splice deltas within a fixed pair of fits
are exact — deterministic forwards over fixed weights — so the numbers in the
table above are resolvable; the floor bounds generalization across seeds, not
the comparison.

Codes are packed sub-byte into uint32 words (9 bits at K=512, 11 at K=2048)
rather than padded to whole bytes, with an fp16 scale per (row, 64 weights).

> **On the "2.7bpw" in the repo name.** 2.635 bpw is the *measured* packed
> figure for the 98.55 GiB base; scaling by the +3.38 GiB of promotions gives
> ~2.73, which is where the name comes from. That is a derivation from two
> measured sizes, not a header pass over the shipped mix — no such pass has
> been run. **The name is a label; the size in GiB is the measured quantity**,
> and the card quotes GiB everywhere a number matters. Kept as 2.7bpw
> deliberately, for consistency with the rest of the lineup's naming.

## Verification and release gate

[TO MEASURE / CONFIRM before push — this section must state what actually ran
on **this** artifact, not on the lineup.]

What is on record for this artifact:

- **It generates.** 19.7 tok/s plain and 19.99 tok/s with the sidecar is real
  generation on the shipping 2.7bpw artifact on the M4. That clears the
  family's standing "nothing has generated a token" blocker.
- **It scores, on all three corpora.** Full referee row measured 2026-09-03
  against the published artifact directory itself, and shown bit-identical to
  the local build it was packed from — so the repack is verified value-neutral.
- `config.json` carries `vision_config` (present via the correspondence
  assertion in the packer, not a presence check) and `image_token_id`.
- The bundle imports only stdlib / mlx / numpy and falls back to
  `mlx_vlm.models.glm5_next`, which is in released mlx-vlm 0.6.17 — audited
  clean of the missing-runtime defect that shipped on three dense 27B
  artifacts.
- Vision provisional cleared by measurement: all 347 vision tensors in the
  struct base are non-zero.

What is **not** on record and gates the push:

- [ ] `check-release` (full gate, including smoke) PASS on this artifact,
      captured with its output. The lineup's 18-artifact static pass was
      `--no-smoke`.
- [ ] `check-bundle` PASS against a **pinned, clean** `vq_switch.py` commit.
      The known runtime-drift blocker is that bundles were built from an
      uncommitted working tree and `check-bundle` passes a dirty tree
      silently. Confirm this artifact's `model.py` provenance before push.
- [x] ~~The two missing referee corpora.~~ **Done 2026-09-03** — code 1.6696,
      literary 1.4843, measured on the published artifact directory.

House rule that produced that list: `smoke` is the only gate that catches an
unexecutable bundle, and it runs last. Two shipped-shaped bundle defects in
this family were caught by luck rather than by a gate.

## Known limitations

- **This is the aggressive end of the ladder.** It beats affine q3 by ~23% KL
  at 27 GiB less, but q3 is a rung we ourselves call collapsed, and prose
  perplexity here is still 1.26x the teacher's. If you have the memory, the
  134 GiB rung reaches a known-usable operating point — and it is not
  published yet.
- **q3 still edges it on literary** (1.4731 vs 1.4843). The one axis where the
  smaller-and-better claim does not hold.
- **Tight on 128 GB with the sidecar.** ~107 GiB resident for trunk + head.
  Cold runs page badly; interleave and discard a cycle before benchmarking.
- **The MTP sidecar is at parity on a single box, not faster** (19.99 vs 19.7
  tok/s). Acceptance is excellent at 0.827; the conversion of acceptance into
  wall-clock is what is missing, and it is a serve-loop problem of ours. No
  single-box speedup is claimed.
- **The VQ switch kernels' small-M behaviour** is a known open frontier for
  this lineup, not a property of the quantization quality.
- **Perplexities here cannot be compared to any other model family** — see
  the contamination note. VQ damage additionally falls hardest on the most
  memorized corpus (literary took the worst absolute damage at 2 bpw and
  recovers the most with bits), so a literary-heavy workload is the one to
  test yourself.
- **The layer-allocation table does not transfer** to other rungs or other
  families. It was measured at this base and is valid only there.
- **MTP requires the absorbed-MLA shim.** Without it the verify step pays a
  23x per-layer attention tax and the sidecar is a net *loss*, not parity.

## Provenance

Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
— **MIT licensed**; see the base model card for the full terms. This is a
quantized derivative and inherits that license. Quantization: TheDrainFlorist,
2026.

The upstream authors ask that their technical report be cited when the model is
used in research:

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

The quantization method — the flat-rung ladder this mix extends, the negative
results, and the measurement rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22136000) (CC BY 4.0).

Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab); the full
experiment log, including what was falsified (the leverage probe, the
"targeting doesn't work" retraction, the dead 80.9 GiB rung) is in the
project's ledger.
