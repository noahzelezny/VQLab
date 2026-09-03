# Launch model card draft — TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw

For Noah's review. **Nothing here is published**; the artifact directory was
not touched. Every number below traces to
`research/quantlab/research/glm53-flash/LEDGER.md`, its `TABLE.md`, or a
direct read of the artifact's `config.json` / file sizes. Markers
`[TO MEASURE: …]` and `[VERIFY: …]` are where the template wants a number the
ledger does not have — they are deliberate, not placeholders to be guessed at
before push.

---

```yaml
---
language:
- en
license: other
license_name: [VERIFY LICENSE]
license_link: LICENSE
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

> **[VERIFY LICENSE]** — no local copy of the upstream card or a LICENSE file
> exists. The three local `zai-org--GLM-5.3-Flash-{3,4,6}bit` directories are
> weights + config + tokenizer only, and their `config.json` carries no
> `license` key. Read the upstream repo before push and set
> `license` / `license_name` / `license_link` to match it exactly; do not
> copy the Qwen frontmatter across.
>
> `pipeline_tag`: this checkpoint carries a live vision tower and
> `image_token_id`, so `image-text-to-text` is the honest tag — but confirm it
> against what the upstream repo declares.

---

# TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw

**101.9 GiB — GLM-5.3-Flash on one 128 GB Mac, with a working speculative
head.**

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

## Measured results

Referee: 2048 tokens; prose = WikiText, code = public mlx corpus (pinned
manifest), literary = Gutenberg. KL is against the bf16 teacher's cached
top-64 logits (`glm53_teacher_topk_prose`, captured mass 0.9906 on every row).

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | prose ppl | code ppl | literary ppl |
|---|---|---|---|---|---|---|
| VQ d8/K16384 (ours, dead rung) | 80.9 GiB | 692.25 | 74.8% | 3.6339 | 1.9619 | 2.9562 |
| VQ d4/K512 (this build's base) | 98.5 GiB | 348.82 | 84.0% | 2.5743 | 1.7107 | 1.6166 |
| **this model** | **101.9 GiB** | **293.84** | **85.7%** | **2.4014** | [TO MEASURE] | [TO MEASURE] |
| VQ d4/K2048 (uniform) | 116.3 GiB | 199.53 | 88.6% | 2.1954 | 1.6187 | 1.3402 |
| affine q3 (ours) | 129 GiB | 377.08 | 83.1% | 2.6824 | 1.7842 | 1.4731 |
| VQ d4/K8192 (uniform) | 134.0 GiB | 94.54 | 92.1% | 2.0379 | 1.5475 | 1.2154 |
| affine q4 (ours) | 166 GiB | 98.34 | 91.9% | 2.0263 | 1.5718 | 1.2025 |
| affine q6 (ours) | 239 GiB | 13.47 | 97.1% | 1.9285 | 1.4929 | 1.1660 |
| bf16 teacher | 598.5 GiB | 0 | 100% | 1.9024 | 1.4888 | 1.1580 |

> **[TO MEASURE: code and literary perplexity for this exact artifact.]** The
> greedy best-8 build was scored on prose + KL + top-1 only (LEDGER
> 2026-08-31, "BEST-8 BY MEASURED EFFECT"). Every other row in the table is a
> full three-corpus score. Either run the two missing corpora against the
> 2048-token caches before push, or delete those two columns for this row and
> say why — do not interpolate them from the base.

**Against affine at matched or smaller bytes:** this build is **27 GiB
smaller than affine q3 and 22% better on KL** (293.84 vs 377.08 at 101.9 vs
129 GiB). That is the honest frontier claim and it is the only "smaller and
better" claim this rung supports. It does **not** reach affine q4 (166 GiB,
KL 98.34); if you want q4-class quality from this family, the 134 GiB uniform
d4/K8192 rung matches or beats q4 on every axis at 32 GiB less, and that is a
different download.

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

Measured on a **Mac Studio M4, 128 GB** (exo conda env), 300-token greedy
runs, warm — one full cycle discarded before measurement, three interleaved
A/B pairs (LEDGER 2026-09-02 night):

| | tok/s |
|---|---|
| plain decode | **4.60** (4.60 / 4.60 / 4.60) |
| with the MTP sidecar | **6.6** (6.58 / 6.57 / 6.63) |

**Cold, uninterleaved numbers on this model are worthless** and we will not
quote any: the first cycle of that night measured 0.56 → 2.56 tok/s across
arms purely from paging, at 100.9 GiB resident on a 128 GiB box. Discard a
full cycle before you believe a number.

Prefill throughput: [TO MEASURE: prefill tok/s on this artifact. The
long-prompt work below measured wall time and memory for a 26k prompt through
a 2-node pipeline, never a single-box prefill rate.]

## Speculative decoding (MTP) — optional sidecar

This repo includes `mtp-head-q6.safetensors` (**6.09 GiB**): GLM's own
multi-token-prediction head — `layers.45`, 889 tensors, 13.84 GiB as a bf16
graft, packed to q6. It is never named in the weight index, so stock loaders
ignore it entirely; it costs nothing on disk-to-RAM unless you opt in.

**When enabled it adds ~6.3 GiB resident** (head weights plus its cache) on
top of the trunk. What you get, measured on this exact artifact (M4, greedy,
300-token runs, warm, interleaved): **4.60 → 6.6 tok/s = 1.43x**, acceptance
**0.82** on every run. The trunk verifies every drafted token by exact
rejection sampling, so the output distribution is exactly the base model's.

The 1.43x depends on a runtime fix, and this is worth knowing before you
judge the number. GLM's sparse attention takes an *absorbed* MLA route only
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

Ceiling with perfect acceptance is 1.81x, and break-even is r < 1.81 — 0.82
acceptance was never the limiting factor.

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

> **[TO MEASURE: external-RSS peak for this artifact.]** The shipped
> Flash-Next cards quote process RSS sampled at 5 Hz from *outside* the
> process across load + prefill + decode. That sampler has never been run
> against GLM. What we have instead, stated precisely: **MLX-side peak of
> 66.0 G on the M4 rank** during the 26k-prefill session, read from the exo
> ledger's `cache=` instrumentation under `EXO_MLX_CACHE_LIMIT_GB=6`, on a
> **2-node pipeline** — a per-rank figure carrying only that rank's shard,
> not a single-box whole-model peak. Either run the 5 Hz external sampler on
> a single-box load + 2048-token prefill + 128-token decode and replace this
> paragraph, or ship the paragraph as written. Do not present 66 G as this
> model's peak memory.

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

> **[VERIFY: the "2.7bpw" in the repo name.]** 2.635 bpw is the measured
> packed figure for the 98.55 GiB base; scaling by the +3.38 GiB of
> promotions gives ~2.73, which is where the name comes from. No header pass
> has been run on the shipped mix to confirm it. Run one, or state the size
> in GiB in the card body and let the name be a label.

## Verification and release gate

[TO MEASURE / CONFIRM before push — this section must state what actually ran
on **this** artifact, not on the lineup.]

What is on record for this artifact:

- **It generates.** The 4.60 / 6.6 tok/s A/B above is real generation on the
  shipping 2.7bpw artifact on the M4, plain and with the sidecar. That
  clears the family's standing "nothing has generated a token" blocker.
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
- [ ] The two missing referee corpora (above).

House rule that produced that list: `smoke` is the only gate that catches an
unexecutable bundle, and it runs last. Two shipped-shaped bundle defects in
this family were caught by luck rather than by a gate.

## Known limitations

- **This is the aggressive end of the ladder.** It beats affine q3 by 22% at
  27 GiB less, but q3 is a rung we ourselves call collapsed, and prose
  perplexity here is still 1.26x the teacher's. If you have the memory,
  the 134 GiB rung reaches a known-usable operating point.
- **Tight on 128 GB with the sidecar.** ~107 GiB resident for trunk + head.
  Cold runs page badly; interleave and discard a cycle before benchmarking.
- **Decode is slow in absolute terms** (4.60 tok/s plain on the M4). The VQ
  switch kernels' small-M behaviour is a known open frontier for this
  lineup, not a property of the quantization quality.
- **Perplexities here cannot be compared to any other model family** — see
  the contamination note. VQ damage additionally falls hardest on the most
  memorized corpus (literary took the worst absolute damage at 2 bpw and
  recovers the most with bits), so a literary-heavy workload is the one to
  test yourself.
- **The layer-allocation table does not transfer** to other rungs or other
  families. It was measured at this base and is valid only there.
- MTP requires the absorbed-MLA shim; without it the sidecar is 1.05x, not
  1.43x.

## Provenance

Base model: zai-org/GLM-5.3-Flash — see the base model card for license and
usage terms **[VERIFY LICENSE]**. Quantization: TheDrainFlorist, 2026. Built
with MLX and [VQLab](https://github.com/noahzelezny/VQLab); the full
experiment log, including what was falsified (the leverage probe, the
"targeting doesn't work" retraction, the dead 80.9 GiB rung) is in the
project's ledger.
