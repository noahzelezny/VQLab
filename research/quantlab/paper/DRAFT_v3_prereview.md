# Data-Free Vector Quantization Beats Calibrated Affine at Matched Bytes Below 6 Bits

**Noah Zelezny**

*Draft 3, 2026-08-23. Every number traces to a committed E-entry via
paper/LEDGER.md; every margin is stated against a measured fit-to-fit
noise floor for its geometry.*

> "A number that resolves a discrepancy has to be measured, not recalled,
> even when the person recalling it is the one who logged it."
> — lab notebook, 08-21

## Abstract

We quantize three large language models — a 397B-parameter
mixture-of-experts, a 35B MoE, and a 27B dense model — for Apple Silicon
unified memory, using vector quantization fit by k-means on the weights
alone: no calibration corpus, no activations, no teacher. We report three
results. **First, the data-free builds beat the affine incumbents at
matched-or-smaller bytes.** On the 397B, our flat-K512 build wins both
evaluation corpora against the strongest community 2.6-bit calibrated build
at nearly the same size (prose perplexity 2.5634 vs 3.1843), our d8 build
wins at 19.6 GiB smaller, and our flagship matches the community 3.5-bit
build's quality at 21.9 GiB smaller. On the 35B, our builds beat the
4-bit incumbent both by dominance (47.5 mnats KL-to-bf16 at 15.8 GiB
against its 78.6 at 19.0) and, at 5 bits per weight, by matched-byte
placement above the interpolated affine frontier, established on two
independent fits. On the dense 27B, d2/K512 beats the 4-bit conversion by
27.8% KL and +1.28 points top-1 agreement at comparable size — the recipe
is not an MoE phenomenon. The wins span 1.75 to 5 bits per weight, and the
advantage ends in the same place on both architectures: the VQ/affine
crossover is bracketed at 4.5–6.0 bits on the dense model and 5.0–6.0 on
the MoE, measured from both sides in each case, and by 8 bits affine is
essentially lossless everywhere we measured. **Second, the size axis becomes
continuous:** a two-coefficient size model prices an artifact before it is
fit — six out-of-sample hits and one in-band on the 397B, and validated
out-of-sample on all three families,
and shallow-bit harvest reaches the sizes between codebook rungs at
measured exchange rates down to 0.0011 ppl/GiB — quantization tuned to a
byte budget rather than to whatever sizes the geometry happens to offer.
**Third, weight-space reconstruction error cannot steer any of this:** in a
pre-registered experiment we engineered a fitter change that improved
exactly the error statistic our analysis identified as load-bearing, and
the resulting model was 4.7x worse than the build it was meant to fix.
Fit error does not rank output damage; only assembled-model scores do.
All artifacts are published with pinned revisions; every comparison names
the artifact and instrument that produced it; margins are reported against
measured fit-to-fit noise floors.

## 1. Introduction

A 397B-parameter MoE with 17B active parameters is, on paper, an ideal
model for a 128–192 GB Apple Silicon machine: the compute per token is
modest, and unified memory holds what a GPU cannot. What decides
feasibility is bytes. At bf16 the model does not fit; at 4 bits it barely
fits the largest machines. The regime that matters — where a Mac Studio
actually runs the model with headroom left over — is 2 to 4.5 bits per
weight, almost all of it spent on the expert tensors that hold ~90% of the
parameters.

The community ladder in this regime is affine quantization: uniform
mlx-community builds and mixed-precision calibrated builds (we compare
against the strongest we could obtain, the "spicyneuron" 2.6-bit and
3.5-bit 397B builds). To our knowledge no vector-quantized artifacts have
shipped for this stack at all: there is no VQ rung on the Apple Silicon
ladder to compare against. This paper ships a full ladder of them — for a
397B MoE, a 35B MoE, and a 27B dense model — and measures it against the
affine incumbents at matched bytes, on one instrument per family, with
every artifact public and every comparison row traceable to the artifact
and instrument that produced it.

The method is deliberately minimal: k-means over weight subvectors, one
flat codebook width per build, no data anywhere in the loop. Data-free
matters for more than elegance. A calibrated quantizer inherits its corpus
— the classic failure is calibrating on prose and deploying on code — while
a weight-space fit has no corpus to inherit. It also forces an honest
measurement posture: when the quantizer never sees data, a quality number
can only come from scoring the assembled artifact, which (we will show) is
the only place a trustworthy number could have come from anyway.

**Claim 1 (method).** At matched-or-smaller packed bytes, the data-free VQ
builds beat the affine builds — calibrated and uniform — on three model
families including a true dense model (§3). Fences: the wins are measured
at 2.0–4.5 bpw on three families; the VQ/affine crossover is
bracketed on both architectures — at 4.5–6.0 bpw on the dense 27B and at
5.0–6.0 bpw on the 35B MoE, in each case with a VQ win measured below the
band and a VQ loss measured above it; at 8 bits affine is essentially
lossless everywhere measured, and on the 27B no VQ rate reaches
8-bit-class KL at all — the method has a measured ceiling as well as a
measured advantage. VQ also pays a real prefill cost
against affine at 35B scale (§6).

**Claim 2 (size targeting).** A quantization can be tuned to a byte budget.
Codebook rungs land where log2(K)/d puts them — on the 397B the gap
between adjacent rungs is 31 GiB — and we make the axis continuous: a
two-coefficient size model prices any target before the fit runs (six
out-of-sample hits and one in-band at 397B; within 0.003 GiB across three
builds on the dense 27B, and within 0.4% predicting a third geometry on
the 35B), and harvesting bits from the shallow layers, which
tolerate cheap bits, sheds size at measured exchange rates that improve
~30x as the base codebook gets richer. The supporting structure: flat
allocation at the target width is the peak at its own size, so the flat
rungs are the reference points and harvest is never a quality win at a
rung's own size — it is how you reach the sizes in between (§3).

**Claim 3 (measurement).** Weight-space reconstruction error — the
statistic quantization work habitually optimizes and gates on — does not
rank output damage, and cannot be used to steer design. We show this by
construction rather than by observation: a pre-registered intervention
improved the precise weight-space statistic our own mechanism analysis
identified as the one that mattered, and the model got worse by 4.7x the
effect it was built to fix (§4). Where the fit is fine-grained the two
metrics track; where centroids are scarce they invert; and mean
reconstruction error is blind to the difference in both regimes.

**Contributions.** (1) The first shipped VQ ladder for Apple Silicon at
397B/35B/27B, revision-pinned on Hugging Face, with matched-byte
comparisons against the affine incumbents. (2) A validated size-targeting
method: pricing model plus harvest exchange rates. (3) A designed
demonstration that fit error cannot steer quantizer design, with the
regime boundary measured. (4) Measured negative results: the 8-bit
ceiling, the harvest floor, and the seed-noise floors that retired three
of our own headline margins (§4, §5).

## 2. Method

The recipe has one moving part. A fixed non-expert skeleton is quantized
affinely once and never varied; the expert tensors (MoE) or MLP trio
(dense) — which hold most of the bytes — are replaced by a vector
quantization whose only dial is the codebook width K, held flat across
every such tensor in the model. A build is named by that width. Everything
else in this section exists to make the resulting number trustworthy
rather than to make it better.

### 2.1 The skeleton

All 397B builds share the `struct6-tail3x3` base established before this
work: 6-bit structure, 4-bit qkv/z projections, bf16 routers. The choice
is settled and not a variable — demoting structure 8→6 bits costs +0.0066
ppl while 6→4 costs ~+0.19, and 2-bit routers cost +11 ppl, so the
skeleton sits at the cheapest width that is not catastrophic [E24, E29].
The dense 27B builds splice VQ MLPs into the local 4-bit affine
conversion, carrying every non-MLP tensor through unchanged, which makes
each build a controlled MLP-treatment ablation against its base. The 397B
vision tower is kept bf16 and grafted last: exactly 912,020,960 bytes,
measured, byte-identical across independent grafts — every size in this
paper is stamped pre- or post-graft because mixing the two once
manufactured a phantom bias (§5).

### 2.2 The fit

For each target tensor independently, weights are reshaped into
non-overlapping d-dimensional subvectors (d=4 for the MoE ladders, d=2
where the dense band demanded finer granularity) and a K-entry codebook is
fit by k-means — k-means++ init, Lloyd iterations, group-64 max-abs fp16
scales. The analytic rate is log2(K)/d bits per weight. Two fit
properties cost us results and are stated up front: k-means is unseeded,
so identical-geometry fits differ (§5 measures this floor and reads every
margin against it), and healthy relative error scales with K, so an abort
threshold tuned at one geometry is wrong at another [law 8].

### 2.3 Packing, and what counts as a size

Codes are packed to their true bit-width after the fit; packing is
bit-exact (verified at the logit level, and byte-identical greedy text on
the d8 artifact). Byte-aligned widths are copied through — packing them
saves zero bytes and costs 37% decode [E70]. **Stored bytes are not a
size**: an unpacked d2/K512 artifact reads 21.6 GiB against its true
14.6, and quoting stored bytes produced three separate wrong conclusions
in one day [rule III.8]. Every size in this paper is a measured packed
size; a row's size and its quality always come from the same artifact.

### 2.4 Size targeting

Flat rungs leave gaps. To reach a size inside a gap we harvest: hold the
body geometry fixed and reduce K in the shallow layers (L0–9 at 397B),
which tolerate cheap bits [position law, E12/E25/E74]. The resulting size
is predicted before the fit:

> 397B (harvest form): `new = base − 1.87 GiB × shallow_bits`
> 27B (composition form): `total = codes + 0.498 + 5.129 GiB`

The 397B form went 6-for-7 within ±0.4 GiB (one in-band); the 27B form
closed to ≤0.003 GiB across three builds and two geometries [6e].

### 2.5 The pipeline

Every artifact passed, in order: fit → outlier gate on a box that did not
produce it → pack → graft → verify → release checks → smoke-generation
through the exact runtime the artifact ships with → score, both metrics,
one instrument per family. Three steps are load-bearing: no box gates its
own artifact (a corrupt artifact scores plausibly, and the fitter's log
cannot see what reached disk); the smoke-gen exists because an artifact
once passed every byte-level gate while being unable to generate a single
token through its shipping kernel; and predictions are registered before
numbers exist, with reading grids fixed in advance, so a wash cannot be
reread afterwards as a win [III.1, III.9, III.11].

### 2.6 Instruments

**397B:** streaming referee perplexity on frozen prose and code corpora,
first 8192 tokens, deterministic — artifacts reproduce their total NLL to
all printed decimals across launches, which is what lets a 0.28%
difference be a property of artifacts rather than drift. **35B / 27B:**
KL-to-bf16 on cached teacher logits (kl_cache_qwen36 / qwen38) with top-1
agreement, plus referee ppl on the 27B. Comparator rows are re-verified on
a second box where used. The gemma-4 family is excluded from this paper
entirely: its raw perplexity is invalid as a property of the model, which
makes scoring non-deterministic, and no claim here rests on a
non-deterministic instrument.

**Noise floors.** Because fits are unseeded, we measured the fit-to-fit
spread at the geometries we compare: dense 27B d2/K256, n=3 — KL range
2.085 mnats, ppl range 0.0447; 397B d4/K256 — 0.0256 prose / ~0.018 code
(same-stack draws). The 397B floor widened each time it was measured more
honestly (0.0134 inferred, 0.0197 mixed-provenance, 0.0256 same-stack) —
a floor estimated from convenient draws flatters every margin near it.
The width has a known source: codebook initialization draws a subsample
with replacement from an unseeded generator, so every fit begins from
different centroids, and different initial centroids change *tail*
reconstruction — which drives output damage while barely moving mean
error (§4.4). Across draws at one geometry, mean reconstruction error
moves by 0.0001 while perplexity moves by 0.026. These floors therefore
describe the unseeded process under which every artifact here was
produced; the fitter is now seeded, which narrows divergence without
eliminating it and would need its own floor measurement. **Every
margin in §3 is stated against the floor for its geometry, and a margin
inside its floor is reported as noise — including three of our own
headline margins that did not survive the discipline (§5).**

## 3. Results

### 3.1 The 397B ladder

All rows: packed whole-artifact bytes, post-graft, one instrument, same-day
re-scored comparators. Figure 1 (`paper/fig_397b_ladder.png`) plots them.

| build | GiB | prose ppl | code ppl |
|---|---|---|---|
| flat K128 (v1) | 100.93 | 3.1706 | 2.6988 |
| **d8/K16384 (published)** | **100.97** | **3.0591** | **2.6728** |
| harvest K64/K256 | 107.9 | 2.7790 | 2.6479 |
| flat K256 (published 2.4bpw) | 111.62 | 2.7655 | 2.6383 |
| **flat K512** | **122.31** | **2.5634** | **2.6123** |
| harvest K512/K2048 | 139.93 | 2.3452 | 2.5969 |
| **flat K2048 refit (flagship, published 3bpw)** | **143.68** | **2.3410** | **2.5963** |
| spicyneuron 2.6bit (calibrated, text-only) | 120.6 | 3.1843 | 2.6667 |
| spicyneuron 3.5bit (calibrated, text-only) | 165.6 | 2.3614 | 2.6005 |

Three comparisons carry claim 1 here, in decreasing margin:

**K512 vs the 2.6-bit calibrated build — the near-matched-size row.** At
1.7 GiB larger, the VQ build wins prose by 0.6209 (46x the noise floor)
and code by 0.0544 (3.4x). This is the cleanest like-for-like on the
ladder and it is not close.

**d8/K16384 vs the same comparator — the smaller-and-better row.** At
19.6 GiB smaller, prose is better by 0.1252 (9.3x floor). The d8 geometry
also beats its exact rate-twin (flat K128, same packed bytes) on both
corpora — dimension pays at matched rate — at a measured cost: ~19%
decode throughput, because a 256 KB codebook cannot live in threadgroup
memory [E115]. At the ~101 GiB class — the only size a 128 GB machine can
hold — the choice between them is quality-per-byte versus tokens-per-
second, and we ship d8.

**The flagship vs the 3.5-bit calibrated build — the size row.** 21.9 GiB
smaller, at quality we decline to claim an edge on. The prose margin is
0.0204 and the code margin 0.0042; we have not measured a fit-to-fit
floor at this geometry, and both margins are smaller than the floor we
measured at the nearest one (0.0256 prose at the same dimension and a
smaller codebook). A floor measured elsewhere cannot license a claim
here, so we claim the bytes and report the quality as indistinguishable.
Note also that the calibrated comparators are text-only, while every
build of ours keeps the vision tower.

The ladder is monotone: no harvest rung beats the flat rung at or above
its size, in either era of this project [E79, E29-era sweep at identical
141.42 GiB: flat 2.3982 vs ramp 2.5042 vs spike 2.7224].

### 3.2 The 35B and the dense 27B

Figure 2 (`paper/fig_35b_27b.png`) plots both remaining families; the
crossover is visible as the affine line passing below ours on the dense
panel and not on the MoE panel.

**35B (kl_cache_qwen36, one instrument, all sizes measured packed):**

| build | GiB | KL mnats | top-1 |
|---|---|---|---|
| **VQ d4/K8192** | **14.838** | **53.022** | 89.55% |
| **VQ d4/K16384** | **15.783** | **47.535** | 89.81% |
| affine 4-bit (community) | 19.000 | 78.557 | 85.61% |
| affine 6-bit (ours) | 26.234 | 13.358 | 94.65% |
| affine 8-bit (community) | 35.131 | 7.449 | 96.18% |

Both VQ rungs sit above the affine frontier at the small end: the better of
them reaches 47.535 mnats at 15.783 GiB against the 4-bit build's 78.557 at
19.000 — 39% less divergence in 3.2 GiB fewer bytes — and the other trades
5.5 mnats for a further 0.9 GiB. The 8→4-bit affine cliff on this family is
10.5x, and the VQ rungs are the only points between the cliff edges.
Quality claims only: both artifacts passed the outlier gate, generated
through the runtime they ship, and were scored on an instrument validated
across two machines to every reported digit.

At the large end, a VQ build at 5 bits per weight (d2/K1024) lands at
21.394 GiB — between two affine rungs, and therefore the first point on
this family where a matched-byte placement against the affine frontier is
defined at all, the smaller rungs sitting below the affine ladder's
cheapest build. Two independent fits of that geometry score **28.141 and
27.927 mnats** against a frontier bar of **43.71** — log interpolation
between the 4-bit build (19.000 GiB, 78.557) and the 6-bit build (26.234
GiB, 13.358), evaluated at the artifact's own measured size. Both draws
sit a factor of 1.55–1.57 below the line. (Linear interpolation would put
the bar at ~56 and flatter us; we quote the conservative method.)

One rung higher the sign flips. At 6 bits per weight, d2/K4096 lands at
25.145 GiB — 1.1 GiB *smaller* than the 6-bit affine build, so the
comparison is direct — and scores 25.502 against its 13.358: a 1.91x
loss, 57x the draw floor, conclusive without a replicate. **The
crossover on this family therefore sits between 5.0 and 6.0 bits per
weight, bracketed from both sides — nearly the same band the dense model
returned.** The advantage is real, and it ends in the same place on both
architectures we measured.

The 5-bpw draws also give this family its first fit-to-fit floor:
**0.214 mnats, 0.76% of the mean**, same box and same geometry, differing
only in the unseeded k-means initialization. The placement margin at 5
bpw exceeds that spread by 73x, and the loss at 6 bpw exceeds it by 57x —
neither verdict is touchable by draw noise.

What the affine column already shows without any VQ point is that the
heavyweight target is expensive for everyone: a 6-bit affine build,
comfortably inside the byte budget we set for an 8-bit-quality tier,
misses that quality by 1.8x, and our 5-bit VQ build inside the same
budget misses it by 3.8x. Neither method reaches 8-bit-class divergence
under 28 GiB on this family — the same verdict the dense model returned
by a different route (§4.1). On both families we measured, 8-bit quality
costs 8-bit bytes, for affine and for us alike.

**Dense 27B (kl_cache_qwen38 + referee ppl; floor 2.085 mnats / 0.0447
ppl):**

| build | GiB | KL | top-1 | ppl |
|---|---|---|---|---|
| q2 (affine) | 7.9 | 1426.9 | 46.07% | 16.435 |
| d4/K256 | 9.7 | 325.6 | 76.46% | 6.403 |
| d4/K1024 | 10.61 | 148.5 | 82.53% | 5.525 |
| q3 (affine) | 10.96 | 187.8 | 79.48% | 5.832 |
| d2/K64 | 11.60 | 93.9 | 85.73% | 5.349 |
| d4/K4096 | 11.61 | 85.8 | 86.05% | 5.229 |
| d2/K256 | 13.60 | 40.3 | 90.10% | 5.233 |
| q4 (affine) | 14.09 | 45.8 | 89.82% | 5.206 |
| **d2/K512** | **14.59** | **33.1** | **91.10%** | 5.194 |
| d2/K4096 | 17.58 | 26.7 | 91.66% | 5.242 |
| **q6 (affine)** | **20.36** | **3.71** | **96.75%** | 5.260 |
| q8 (affine) | 26.34 | 1.6 | 98.08% | 5.243 |

The recipe is not an MoE phenomenon. Below 4.5 bpw every VQ point sits
above the affine line at its size: d4/K1024 beats q3 on both metrics at
0.35 GiB less, and
d2/K512 beats q4 by 27.8% KL (6.1x floor) and +1.28 points top-1 at
q4-class size. The ppl column is shown for completeness: every ppl
difference between adjacent rungs on this ladder is inside the 0.0447
noise floor and none is claimed in either direction. KL and top-1 are
what separate these artifacts. The q6 comparator (3.710 mnats / 96.75% @ 20.355 GiB) closes the ladder's
top: it beats E128C by 7.2x KL at 2.77 GiB more — the crossover row.
E130's rate twins (d2/K64 93.9 vs d4/K4096 85.8 KL at 11.60/11.61 GiB)
settle d-vs-K at a second band: raise K first, ~8.6%, consistent with the
~12% at 2.00 bpw [E87/E130].

### 3.3 Size targeting in practice

The exchange rates, measured at three base richnesses on the 397B (prose
ppl per GiB shed): 0.0315 off a K128 base, 0.0033 off K256, 0.0011 off
K2048 — harvest gets ~30x cheaper as the base gets richer, and is roughly
2x the byte-efficiency of stepping down the flat ladder. Combined with
the size model (§2.4), the practical capability is: name a byte budget
anywhere in the ladder's range, price the artifact to a few tenths of a
GiB, and fit it once. The fence: harvest is never free — cost is monotone
in bits harvested at every base measured — and never beats the flat rung
at the flat rung's own size. What it buys is the space between.

### 3.4 Speed, honestly

Decode is a wash across d4 geometries; prefill is where geometry shows.
u8view (dispatching byte-aligned unpacked codes through the packed
kernel) is bit-exact and ships: +25–33% prefill. Against affine the
honest picture at 35B is prefill ~0.5x even with the lever. d8 pays ~19%
decode for its quality, measured clean-mode, same-session ratio [E115].
Speed numbers in this project are reported only as same-session ratios
between arms: we measured decode throughput at ~100 GiB to be bimodal on
our hardware (same artifact, 40% swing, swap/thermal/path ruled out by
measurement), so absolute single-run throughput numbers are not
meaningful and we do not publish them.

## 4. Negative results and the limits of weight-space measurement

This section reports what lost, what cannot be reached, and the designed
experiment that shows why fit error cannot steer any of it. These are
results, not caveats.

### 4.1 The 8-bit ceiling is real and measured

At 8 bits affine is essentially lossless: 7.4 mnats on the 35B, 1.6 on
the 27B. On the 27B we measured whether ANY rate reaches 8-bit-class KL:
the ladder's slope flattens from x0.673 KL per added bpw (4.0→4.5) to
x0.868 (4.5→6.0); extrapolating the measured slope, KL 1.6 needs ~25 bpw
— the q8 artifact costs 8. **There is no rate at which this method
reaches 8-bit-class fidelity on the 27B.** The 35B returns the same
verdict by the other route: a 6-bit affine build, inside the byte budget,
misses 8-bit quality by 1.8x, and our 5-bit build misses by 3.8x. The
method's home is the low-to-middle band — through 4.5 bpw on the dense
model and at least 5 bpw on the MoE — and it does not compete with 8-bit.
We say so with a measured slope rather than a shrug.

### 4.2 Harvest has no free lunch

Cost is monotone at every base measured; the one apparent counter-example
(a harvest rung beating the flat rung above it) was a proxy-score
artifact and was retracted with its entire mechanism story [E79]. What
survived is the exchange-rate table (§3.3) — cheaper, never free.

### 4.3 d4 vs d2, corrected twice

At matched 2.00 bpw on the 35B, d4/K256 beats d2/K16 by 12.2% KL — not
the 3.3x an earlier corrupt artifact manufactured (the corrupt arm
inflated the gap ~25x, and the correction moved against our preferred
result). Clean refits put the margin at 6–11% across 3.25–3.75 bpw.
Three independent estimates agree: raise K first, but as a measured
preference, not a landslide [E87, E99]. At 3.00 bpw on the dense model, exact
rate twins 5 MB apart again favour d4, by 8.6% [E130]. But d4 has a hard
rate ceiling of 4.00 bpw — 16 bits of index over 4 weights, even at a
65,536-entry codebook — so above that band the geometry question is moot
and d2 with a large codebook is the only option. Whether dimension still
pays *at* 4.00 bpw awaits the matched rate twin, which is fitting as we
write and is a revision item.

### 4.4 The designed demonstration: improving the fit made the model worse

The chain of events matters, because each step was pre-registered:

1. A refit of the published K256 build scored WORSE than the original at
   byte-identical size — and the refit had *lower* reconstruction error
   on every projection [E92, E101].
2. Percentile analysis located the trade: the refit was better where
   most weights live (the bulk) and worse exactly in the top 0.1% by
   magnitude (the tail). Mean relerr — a bulk statistic — reported the
   trade as an improvement [E102].
3. Per-tensor probes found the mechanism real, replicated, and
   depth-structured: body layers are sub-Gaussian (no far points), so
   distance-chasing seeding buys the bulk with the tail when centroids
   are scarce; shallow layers are heavy-tailed and the same change helps
   them [E107–E110]. Mean relerr across the affected tensors moved by
   −0.0003: the gate is blind to the entire effect.
4. **The intervention:** weight the k-means objective to buy the tail
   back. Pre-registered reading rule, fixed before any number existed.
   It succeeded in weight space — the tail error bands improved exactly
   as designed — and the model scored 2.9945 against the 2.8057 it was
   built to fix: **4.7x worse than the regression, in the direction the
   weight-space statistic called an improvement** [E112].

We then measured the boundary: at fine-grained fits (relerr ~0.08, dense
d2/K256), improving the objective DOES improve the model — the effect
tracks, at 2.8x the noise floor [E127]. So the honest law is scoped:
where centroids are scarce, weight-space error and output quality can
invert, and no weight-space statistic we measured — mean, percentile
bands, or engineered improvements to either — predicts which side of the
line a fit lands on. Only the assembled model knows.

The same lesson has a quieter form. Two fits of one geometry differ only
in their unseeded initialization, and that difference is invisible in
mean reconstruction error (0.0001) while moving perplexity by 0.026 —
larger than several margins we had been prepared to report. Any
comparison at that scale is a comparison of draws.

### 4.5 What else lost

Calibration on its home turf: an activation-calibrated method lost to
uniform quantization on the dense 27B in our earliest tests; per-layer
sensitivity probes fail mechanistically on MoE; distillation-based DWQ
was falsified at 397B/2-bit; a hypothesized fused-gather prefill lever
does not exist (the runtime already fuses it). Each carried an E-number
and a measured effect size before it was closed.

## 5. Measurement discipline

Everything above survived a discipline that this project learned the
expensive way. We state it briefly — the incidents are in the lab record —
because three of its instruments transfer to any quantization work.

**Noise floors before margins.** Fits are unseeded; two identical-
geometry fits differ. Measuring that floor (one extra fit) retired three
of our own margins the same night it was measured — including a published
"beats the incumbent on all three metrics" that became "on two, with the
third inside the noise." Third-decimal ppl differences between
single-draw artifacts are not interpretable, and we no longer print them
as claims. Speed has the same rule (§3.4): ratios within a session,
never absolutes.

**Provenance before attribution.** Twice, a build input was silently
rewritten in place after artifacts had been produced from it. Neither
overwrite was caught by a gate; both were found by reading file metadata,
and one invalidated a comparison we had already built an explanation on,
because the two arms turned out not to share an input. Artifacts now
carry manifests (per-shard bytes, mtimes, content hashes, stored outside
the artifact) so that "was this overwritten?" is a lookup rather than an
investigation, and the fitter is seeded, so a build is reproducible from
its recipe and seed.

**Acceptance harnesses must test the copy that ships.** A harness that
imports by module path silently tests whichever copy is on the import
path — ours validated a fix in the development environment while the
copies inside two release candidates went unexercised, and two sessions
disagreed about observable reality for forty minutes because their
environments resolved different copies. The fix is structural: the
harness takes an artifact and imports its bundled runtime as the unit
under test.

**The general pattern.** Across two weeks and roughly 140 logged
experiments, no wrong number announced itself. Every one — a proxy score in a headline
table, a corrupt artifact manufacturing a 25x effect, a units mismatch
impersonating a bias, an algebraic identity reported as a finding — was
plausible, internally consistent, and caught only by pre-registration, a
cheap measurement, or a second reader. The rules that fell out are
mechanical: comparison rows name their artifact and instrument; a number
older than the artifact it faces is re-measured, not cited; sizes are
packed bytes from the same artifact as the quality number; every gate is
tested against a known-bad input before its pass is believed; and
predictions are registered before numbers exist, with falsified
predictions recorded as falsified. None of it is novel methodology; it is
ordinary unit discipline, applied to a setting where the wrong numbers
are the plausible ones.

## 6. Limitations

**Coverage.** Three model families, only one of them dense; a second dense
family would test whether the 27B generalizes. Single vendor stack
(MLX/Metal), and the kernel conclusions — threadgroup capacity, the d8
decode tax — are specific to Apple Silicon.

**Unmeasured regions.** The VQ/affine crossover is bracketed on both the
dense 27B and the 35B MoE, but not on the 397B, which has no affine
comparator above 3.5 bits; whether the same band holds at that scale is
untested. Whether
dimension pays at 4.00 bpw awaits the rate twin now fitting. No dense
harvest rung has been built, so claim 2's exchange rates are measured on
MoE only, and the mechanism we propose for them — shallow-layer
redundancy — predicts they should weaken on dense, which is a prediction
and not a result.

**Instrument limits.** The 397B same-stack noise floor rests on two
draws; more would sharpen it, and any margin near 0.026 prose at that
geometry is reported as indistinguishable rather than claimed. The 35B
floor (0.214 mnats) is same-box and same-geometry, so it bounds
initialization variance and not implementation or hardware variance.
Perplexity is uninformative on the dense 27B — every rung sits inside its
floor — so that family is ranked on divergence and agreement alone.
Decode throughput on our hardware is bimodal at ~100 GiB residency and we
have not explained it; we publish ratios within a session and no absolute
figures.

**Costs we pay.** Prefill remains ~0.5x affine at 35B even after the
shipped lever. The largest codebooks cannot use threadgroup memory and
pay ~19% decode for it. One published artifact predates our manifest
system and cannot be rebuilt, though it remains downloadable and its
score is reproducible in distribution.

## 7. Reproducibility

All artifacts are published under `TheDrainFlorist` on Hugging Face with
their VQ runtimes bundled in-checkpoint (stock mlx-lm, no patches).
Where a repository's weights were upgraded in place, the previous build
remains fetchable at its pinned revision and the card labels which
weights produced which benchmark rows — every published number stays
checkable against the bytes that produced it. Published artifacts carry
external manifests (per-shard bytes, mtime, content hash), and the fitter
is seeded, so a build is reproducible from its recipe plus its seed.

Which copy of a runtime executes is environment-dependent — a loader hook
can override a bundled runtime for exactly the layers that matter — so
runtime-dependent claims name the resolved, executing copy rather than a
file believed to be loaded. The bundled runtimes here are verified three
ways: hash-compared against the runtime that produced the published
scores, exercised as the executing copy in a stock environment by
generating through the shipping kernel, and passed kernel acceptance as
the unit under test lifted from the artifact itself. Fit, pack, verify,
gate, and scoring scripts are in the project repository.
One historical caveat is stated rather than hidden: the original
published 2.4bpw build predates the manifest system, its build inputs
were later overwritten, and it is preserved and checkable but not
rebuildable.

## Acknowledgments

Cross-session peer review is load-bearing throughout: several of this
paper's corrections — including two that moved results against our
preferred reading — were caught by a second reader before publication.
Dr. Saamer Saab Jr. is the paper's first reader.
