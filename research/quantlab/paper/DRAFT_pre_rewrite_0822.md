# [TITLE — see OUTLINE.md candidates; repo name on hold pending E95]

Noah Zelezny
*Draft skeleton, 2026-08-21. Sections 3, 4, 6, 7 are annotated stubs; five
results land today and are held as explicit `[[PENDING]]` slots — nothing
below guesses their outcomes.*

> "A number that resolves a discrepancy has to be measured, not recalled,
> even when the person recalling it is the one who logged it."
> — lab notebook, night of 08-20/21 [E-record, night log 03:35]

> "Accurately describing the wrong outcome is not the same as catching it."
> — lab notebook [[UNVERIFIED — quote not yet located verbatim in the
> E-record; see OUTLINE ambiguity #5]]

## Thesis

In the 2–3.5 bits-per-weight regime where large mixture-of-experts models
actually fit on Apple Silicon, quantization quality is bought by **codebook
richness applied flatly across the expert surface**, and it can be bought
**with no calibration data at all**. Data-free vector quantization — k-means
over weight subvectors, no corpus anywhere in the loop — beats calibrated
affine quantization at matched packed bytes on MoE expert tensors, and the
same recipe places above the affine ladder on a true dense model [E95]. Flat
allocation at the target width is the peak of that ladder; and precisely
because it is the peak, the useful role of mixed allocation is not to beat it
but to **price the sizes between flat rungs**. That yields a second,
practical result: a measured exchange rate and a two-coefficient size model
that together let you name a byte budget and get the best artifact at it in a
single fit, rather than being restricted to the sizes codebook steps happen
to land on.

*Fencing, stated up front: claim 1 lives strictly at 2–3.5 bpw — at 8 bits
affine is essentially lossless and there is nothing to beat. Claim 1 is not
an MoE-expert phenomenon: on a true dense model (Qwen3.8-27B) the same flat
recipe, in its weakest configuration, lands ~26% above the affine ladder at
its size [E95] — the dense K ladder is in progress `[[PENDING: E119]]`.
Claim 2 remains measured on MoE experts only; no dense harvest rung has been
fit. Every claim below is scoped to the instrument that produced it.*

## Abstract

We report two narrow, thoroughly measured results on quantizing
mixture-of-experts (MoE) language models for Apple Silicon unified memory.
**First**, data-free vector quantization — per-tensor k-means on the weights
alone, with no calibration corpus — beats calibrated affine quantization at
matched bytes in the 2–3.5 bits-per-weight MoE-expert regime. On
Qwen3.5-397B-A17B, our flat d4/K2048 build measures wikitext perplexity
2.3410 and code perplexity 2.5963 at 143.65 GiB, beating the community
3.5-bit calibrated build (2.3614) at 21.9 GiB smaller, with the vision
tower intact where the community builds are text-only [E91]; our 100.9 GiB
rung beats the community 2.6-bit at ~20 GiB smaller [E74 refs]. On
Qwen3.6-35B-A3B, our d4/K8192 build reaches 56.4 mnats KL-to-bf16 against
the community 4-bit's 78.6 at smaller size, on one instrument [E84]. The
claim is scoped: at 8 bits the advantage vanishes (8-bit affine is
essentially lossless, 7.4 mnats [E84 addendum]), and whether the recipe
carries to dense models is an open, registered question
`[[PENDING: E95 dense]]`. **Second**, flat codebook allocation at the
target width is the peak of the size-quality ladder: no mixed allocation we
measured beats the flat rung at or above its own size [E79], and a
counter-design that funds a body upgrade from shallow-layer savings is
priced out by the measured 4.7:1 shallow:body byte ratio [E74 addendum].
What mixed allocation buys instead is the space between flat rungs:
"cheap-shallow harvest" sheds bytes at measured exchange rates of 0.0315
ppl/GiB from a K128 base down to 0.0011 ppl/GiB from a K2048 base — about
2x the byte-efficiency of stepping down the ladder — and a two-coefficient
size model prices any target size to within ~0.4 GiB before fitting
(5–6/6 out-of-sample) [E78, E79 addendum, E91]. **Third**, and unusually,
we document our errors as part of the method: a proxy score that poisoned a
headline comparison [E79], a corrupt artifact that manufactured a
25x-overstated effect [E82/E85], and an algebraic identity briefly reported as a finding [E91
correction]. Every one of these wrong numbers looked plausible; each was
caught by pre-registration, cheap measurement, or cross-review, and each
produced a concrete gate now in the pipeline. All artifacts are published
with pinned revisions; every comparison names the artifact and instrument
that produced it.

## 1. Introduction

A 397B-parameter MoE with 17B active parameters is, on paper, an ideal
model for a 128–192 GB Apple Silicon machine: the compute per token is
modest and unified memory holds what a GPU cannot. What decides feasibility
is bytes. At bf16 the model does not fit; at 4 bits it barely fits the
largest machines; the regime that matters — the one where a Mac Studio or a
two-box Thunderbolt cluster actually runs the model with headroom for other
work — is 2 to 3.5 bits per weight, almost all of it spent on the expert
tensors that hold ~90% of the parameters.

The community ladder in this regime is affine quantization: uniform
mlx-community builds and mixed-precision calibrated builds (we compare
throughout against the strongest we could obtain, the "spicyneuron" 2.6-bit
and 3.5-bit 397B builds). To our knowledge no vector-quantized MoE
artifacts have shipped for this stack at all: **there is no VQ rung on the
Apple Silicon ladder to compare against.** This paper ships one — a full
ladder of them — and measures it against the affine incumbents at matched
bytes, on one instrument, with every artifact public and every comparison
row traceable.

Our method is deliberately minimal: k-means over d=4 weight subvectors,
one flat codebook width per build, no calibration data anywhere in the
loop. Data-free matters for more than elegance. A calibrated quantizer
inherits its corpus — the classic failure is calibrating on prose and
deploying on code — while a weight-space fit has no corpus to inherit. It
also removes an entire class of instrument risk we spent much of this
project learning to respect: when the quantizer never sees data, a quality
number can only come from measuring the assembled artifact, which is where
it should come from anyway (weight-space fit error demonstrably does not
rank output damage [E55; FINDINGS law 6]).

We make two claims and fence both.

**Claim 1 (method).** In the 2–3.5 bpw MoE-expert regime, at matched
packed bytes, the data-free VQ builds beat the calibrated affine builds on
both of our corpora, on two model families (§3). The fences: at 8 bits
affine is essentially lossless and there is nothing to beat [E84 addendum];
VQ pays a real prefill cost against affine at 35B scale
(§6); and whether any of this carries to a true dense model is an open
question registered before its experiment ran `[[PENDING: E95 dense]]`.

**Claim 2 (size targeting).** A quantization can be tuned to a particular
size target while retaining quality gracefully. The size axis is normally
discrete — a codebook width lands where log2(K)/d puts it, and at 397B the
gap between two adjacent flat rungs is 31 GiB — so a machine with 120 GiB of
usable memory is offered a build at 112 or one at 143. We show the axis can
be made continuous: holding the body geometry fixed and harvesting bits back
from the shallow layers sheds bytes at a measured exchange rate, falling
from 0.0315 ppl/GiB off a K128 base to 0.0011 ppl/GiB off a K2048 base —
roughly twice the byte-efficiency of stepping down the flat ladder — and a
two-coefficient size model prices the result to within a few tenths of a GiB
*before* the fit is run, with six out-of-sample hits and one in-band, no
misses (§3). Naming a byte budget and getting the best available artifact at
it is the claim; the supporting structure is that flat allocation is the
peak at its own size, so the flat rungs are the reference points the tuning
works between, and harvest is never a quality win at a rung's own size
(§3, §4).

**Contributions.**
1. The first (to our knowledge) shipped VQ MoE artifacts for Apple
   Silicon, with a measured size-quality ladder at 397B and 35B (§3),
   revision-pinned on Hugging Face (§7).
2. The matched-byte comparisons behind claim 1, all same-instrument,
   pre/post-graft stamped, packed-bytes-only (§3).
3. The flat-is-peak result and the harvest exchange-rate table behind
   claim 2, plus a validated two-coefficient size model (§3).
4. A negative-results section reporting what lost, at what effect size,
   including the corrected d2-vs-d4 gap and the retractions (§4).
5. An instrumentation section describing four plausible-looking wrong
   numbers, how each was caught, and the gate each produced (§5). We think
   this section is the most reusable part of the paper.

## 2. Method

The recipe has one moving part. A fixed non-expert skeleton is quantized
affinely once and never varied; the expert tensors — which hold roughly 90%
of the parameters and therefore essentially all of the byte budget — are
replaced by a vector quantization whose only dial is the codebook width K,
held **flat across every expert tensor in the model**. A build is named by
that width. Everything else in this section exists to make the resulting
number trustworthy rather than to make it better.

### 2.1 The skeleton, and why the experts are the surface

All 397B builds share the `struct6-tail3x3` base established in the affine
era: 6-bit structure, 4-bit qkv/z projections, bf16 routers, with the last
three layers' experts promoted [E29 recipe of record]. The choice of
skeleton is settled and not a variable here — demoting structure 8→6 bits
costs +0.0066 ppl while 6→4 costs ~+0.19, and 2-bit routers cost +11 ppl
[E24, E29], so the skeleton sits at the cheapest width that is not
catastrophic and stays there. In the VQ builds the base contributes only
structure, projections and routers; the expert region is discarded and
refit.

The vision tower is kept at bf16 and **grafted** onto the packed artifact as
a final step. It is exactly **912,020,960 bytes (0.849 GiB)** — measured
directly from the shard headers and verified byte-identical across two
independently grafted artifacts [night log 08-21 03:35]. This matters for
reporting, not for quality: it is the difference between a pre-graft and a
post-graft size, and mixing the two once presented as a false "flat-geometry
bias" in our size model (§5.4). Every size in this paper is stamped.

### 2.2 Flat vector quantization of the expert tensors

For each expert tensor independently, weights are reshaped into
non-overlapping **d = 4** subvectors and a **K**-entry codebook is fit by
k-means (k-means++ initialization, Lloyd iterations with scatter-add
centroid updates, group-64 max-abs fp16 scales). No calibration corpus, no
activations, no teacher: the fit sees weights and nothing else. The artifact
stores per-tensor codebooks plus one code index per subvector, so the
analytic rate is log2(K)/d bits per weight — K128 → 1.75, K256 → 2.00,
K512 → 2.25, K2048 → 2.75.

Two properties of the fit are worth stating because both have cost us
results. First, **k-means is unseeded**, so two fits of the same tensor at
the same geometry differ slightly; where that fit noise sits inside a
reported effect we say so (§3, the +0.0042 harvest cost contains four such
tensors). Second, **healthy relative error scales with K** — K2048-class
fits sit near 0.19, K256 near 0.31, K128 near 0.46 — so an abort threshold
tuned at one geometry is wrong at another. A 0.35 threshold carried over
from K2048-era numbers would have aborted a perfectly healthy K256 refit
[law 8; near-miss, night log 08-21]. Abort bars are set per geometry.

### 2.3 Geometry: why d = 4 and why K is the dial

The subvector dimension d and the codebook width K both buy rate, and the
question of which to spend on has a measured answer at one point. At matched
2.00 bpw, with both arms fit fresh from the same bf16 source on the same
box, both outlier-gated before scoring, and both scored on the same
instrument: **d4-K256 = 210.7 mnats KL / 80.05% top-1 versus d2-K16 = 239.9
/ 78.43%** — d4 wins by 12.2% KL and +1.62 points [E87]. The honest scope is
one pair at a bpw that forces d2 to a very coarse K16; this is a measured
preference for raising K, not a general law about dimension. Notably, the
same question was "settled" once before with a 3.3x effect that turned out
to be a corrupt artifact — the real gap is ~25x smaller (§5.2).

Whether the ladder continues upward in d — whether d8 with a
correspondingly larger codebook beats d4 at matched rate — is measured in
this paper with its reading grid fixed in advance `[[PENDING: E89 d8
verdict]]`.

Within d4, **K256 is the operational sweet spot**: its 8-bit codes are
byte-aligned, so it is the only rung that pays no unpacking tax at all, and
its codebook fits in threadgroup memory. Above K256 the codes need uint16,
pay a packing tax, and cost substantially more fit time; a K65536 codebook
(1 MB) exceeds Apple's 32 KB threadgroup memory entirely and has no fast
kernel [E83]. We leave K256 only when the size budget forces us.

### 2.4 Packing, and what counts as a size

Codes are packed to their true bit-width **after** the fit. Packing is a
pure storage transform and is bit-exact: packed artifacts reproduce their
unpacked scores to the full precision of the total negative log-likelihood.
One exception is enforced by the packer: when the code width is already a
multiple of 8, packing saves exactly zero bytes while routing the tensor
through bit-field extraction — a 37% decode regression for no gain — so
byte-aligned widths are copied through [E70 addendum].

**Stored bytes are not a size.** Unpacked artifacts carry whole-byte
padding: d2-K16 occupies 21.33 GiB unpacked and 13.83 GiB packed, which is
exactly d4-K256's size, as matched 2.00 bpw requires. Reading the unpacked
numbers as a size difference produced three separate wrong conclusions in a
single day [E87 correction; rule III.8]. Every size in this paper is a
packed size or an analytic rate, stamped pre- or post-graft.

### 2.5 Size targeting: harvest and the size model

Flat rungs land where log2(K)/d puts them, which leaves large gaps — 31 GiB
between K256 and K2048 at 397B. To reach a size inside a gap we **harvest**:
hold the body geometry fixed and reduce K in the shallow region (layers
0–9), which the position law identifies as the part of the network that
tolerates cheap bits [law 2]. This is a subtraction, not a reallocation —
the bytes are given back, not moved elsewhere (§3 explains why moving them
is not worth doing).

The resulting size is predicted before the fit by a two-coefficient model
measured on the 397B:

> `new_size = base_size − 1.87 GiB × shallow_bits_harvested`

with the shallow region L0–9 costing 1.87 GiB per bit and the body L10–56
costing 8.81 GiB per bit. Its out-of-sample record is reported in §3.

### 2.6 The pipeline and its gates

Every artifact in this paper passed the same sequence, in order:

**fit → outlier gate on a trusted box → pack → graft → verify →
`check_release` / `check_bundle` → referee, both corpora, serially.**

Three parts of that chain are load-bearing rather than ceremonial.
**Fits and gates run on different machines**: fitting is distributed across
two boxes, but no box gates its own artifact, because one of them has
produced artifacts that passed the fitter's own log and were corrupt on disk
[E47]. **The outlier gate runs before scoring, always** — a corrupt artifact
scores plausibly and silently, and the fitter's log structurally cannot see
the corruption because it reports what it computed, not what reached disk
[rule III.9; §5.2]. **Predictions are registered before the number exists**,
with a reading grid that names what each outcome would mean, so that a wash
cannot be read afterwards as a failure or a win [rule III.1].

### 2.7 Instruments

Two, and no result mixes them.

**397B — referee perplexity, prose and code.** Raw perplexity on the first
8192 tokens of a frozen referee corpus, growing context, streamed so that a
model larger than RAM can be scored. It is deterministic: artifacts
reproduce their scores to the exact total negative log-likelihood across
launches and across sessions, which is what allows a 0.28% difference
between two builds to be read as a property of the artifacts rather than
drift [E80]. Both corpora are reported for every 397B row; where they
disagree, we say so.

**35B — KL to bf16.** Mean KL divergence against cached bf16 teacher logits
over a fixed 8192-token window (`kl_cache_qwen36`), reported in mnats
alongside top-1 agreement. Comparator rows were re-scored independently on
two boxes and agree digit-for-digit [E84 addendum].

Two instrument decisions are worth stating because they exclude evidence
that would otherwise have been convenient. **Absolute KL is not comparable
across teacher bases**: the clean d2/d4 pair was fit fresh from bf16 with
non-expert tensors left at bf16, so its absolute values do not sit on the
same scale as the struct6 rungs, and only the within-pair difference is a
result [E86]. And **the gemma-4 family is excluded from this paper
entirely**: its perplexity is invalid as a property of the model rather than
of our harness, which makes scoring non-deterministic, and no quality claim
here rests on a non-deterministic instrument.

Finally, a comparison row is only believed if it names the artifact and the
instrument that produced it, and any number older than the artifact it faces
is re-measured rather than cited [rule III.2]. That rule exists because
violating it once cost a day of mechanism work on an anomaly that did not
exist (§5.1).

## 3. Results — the ladders — [TABLES CURRENT AS OF 08-21 PM; prose to draft]

*All rows: one instrument, packed whole-artifact bytes, post-graft stamped.*

**397B (referee prose/code ppl):**

| build | GiB | prose | code | source |
|---|---|---|---|---|
| flat K128 (shipped 2.2, old fitter) | 100.930 | 3.1706 | 2.6988 | E74 refs |
| **flat d8-K16384** | **100.970** | **3.0591** | **2.6728** | E92/E93 ladder [verdict entry pending] |
| harvest K64/K256 | 107.9 | 2.7790 | 2.6479 | E79 |
| flat K256 (shipped 2.4, old fitter) | 111.617 | **2.7655** | **2.6383** | E79/E92 |
| flat K256 refit (E92 — regression, not shipped) | 111.617 | 2.8057 | 2.6447 | E92 |
| **flat K512 (new rung)** | **122.305** | **2.5634** | **2.6123** | E93 |
| harvest K512/K2048 (best-per-GiB) | 139.93 | 2.3452 | 2.5969 | E80 |
| flat K2048 (shipped 3.1, old fitter) | 143.682 | 2.3519 | 2.5987 | E91 |
| **flat K2048 refit (flagship)** | **143.682** | **2.3410** | **2.5963** | E91 |
| harvest K64/K128 | 99.05 | 3.2289 | 2.7078 | E78 |
| harvest K32/K128 | 97.2 | 3.2730 | 2.7055 | E74 |
| spicyneuron 2.6bit (affine, blind) | 120.6 | 3.1843 | 2.6667 | E35-era, rescored refs |
| spicyneuron 3.5bit (affine, blind) | 165.6 | 2.3614 | 2.6005 | E91/card G |

Prose notes to draft:
- E93 vs spicy 2.6bit is the near-matched-size showcase: +1.7 GiB, prose
  −19.5%, code −2.0% — the registered "meaningful margin" prediction,
  confirmed.
- The fitter-vintage effect is K-DEPENDENT: −0.0109 at K2048, +0.0402 at
  K256, −6.0% at 35B K8192. Three matched pairs, mechanism unidentified
  [E91/E92/E94]. State scoped to geometry; never as "the improved fitter."
- d8 beats its rate-twin on both corpora at matched size; the scrutiny-bar
  note (3.0591 vs the 3.05 line) is stated, not hidden [E89 grid].
- Harvest exchange rates: 0.0315 / 0.0238 (K128 base), 0.0033 (K256),
  0.0011 (K2048) ppl/GiB vs flat slope 0.0365/0.0129 [E78, E79 add., E91].
- Size model scorecard incl. E92/E93 graft confirmations (111.617 vs
  ~111.62 pred; 122.305 vs ~122.31 pred).
- Chart: regenerate chart_397b_ladder.py with E91/E92/E93/d8 points AND the
  spicy x-coordinate fix (121.0 -> 120.6).

**35B (kl_cache_qwen36, mnats / top-1, all outlier-gated per E97/E98/E99):**

| build | bpw | KL mnats | top-1 |
|---|---|---|---|
| mlx 8-bit | 8 | 7.449 | 96.18% |
| **d4-K8192 refresh (E94)** | 3.25 | **53.022** | **89.55%** |
| d4-K8192 (standing) | 3.25 | 56.413 | 89.37% |
| mlx 4-bit | 4 | 78.557 | 85.61% |
| d4-K4096 | 3.00 | 68.546 | 87.9% |
| d4-K2048 | 2.75 | 85.535 | 87.3% |
| d2-K256 | 4.25 | 36.862 | 90.9% |
| d2-K128 refit | 3.75 | 49.984 | — |
| d2-K64 refit | 3.25 | 73.259 | — |
| d4-K256 | 2.00 | 210.7 | 80.05% |
| d2-K16 | 2.00 | 239.9 | 78.43% |

- The 8->4-bit cliff is 10.5x; d4-K8192 is the only point between the cliff
  edges — better AND smaller than the community 4-bit.
- Clean d2-vs-d4 margins: 6.4% (3.25 bpw), 11.4% (3.75, vs interp), ~12%
  (2.00, E87). The corrupt arm's 3.3x is void; the correction moved AGAINST
  our preferred result [E99].

**Speed:** decode is a wash across geometries; prefill is where geometry
shows [E70/E71/law 9]; u8view +25–33% prefill, bit-exact, shipped [E81,
E90]; honest 35B picture: VQ prefill ~0.5x affine even with the lever
[E81].

**Dense:** `[[PENDING: E95]]` — reported here if positive, §4 if negative.
Placement-on-ladder reading only; never "beats 4-bit" (not size-matched).

## 4. Negative results — [STUB]

*To draft; each with effect size and E-citation. This section is
load-bearing, not an appendix.*
- d2-vs-d4 at matched 2.00 bpw: d4 wins by 12.2% KL / +1.62 top-1 —
  and packed sizes are IDENTICAL (13.83 GiB both); the corrupt-arm 3.3x
  is void [E87 + correction, E85].
- Harvest is never free: monotone cost at every base; no floor above zero
  [E78, E79]. The retracted "cheap-shallow beats the rung above" story and
  why it existed (proxy score) [E79].
- Geometry washes within d4 at matched bytes; across-d is measured
  non-wash at 2.0 bpw only [E84, law 1 scope].
- Calibration losing on its home turf: OptiQ calibrated < uniform on dense
  27B [E40/E42]; per-layer sensitivity probes mechanistically wrong on MoE
  [E7]; DWQ falsified at 397B/2-bit [E27/E28]; fused row-gather prefill
  lever does not exist [E52].
- `[[PENDING: E95 dense]]`: registered — if VQ loses on true dense, that
  is reported here as the measured boundary of claim 1.

## 5. How every wrong number looked plausible: instrumentation

The results above passed through a pipeline that, by the end of this
project, contained nine written instrument rules and six executable gates.
None of them were designed in advance. Every one exists because a specific
wrong number — plausible, internally consistent, and in two cases already
in a headline table — survived until something cheap and mechanical caught
it. This section reports those incidents in the order they taught us,
because we believe the pattern generalizes: **in this work, no wrong result
announced itself. Each was caught by pre-registration, a cheap measurement,
or a second reader — never by inspection of the number itself.**

**5.1 The proxy score that poisoned a headline (→ comparison-row rule,
`check_comparator.py`).** For most of a day the notebook's central result
was that a 107.9 GiB mixed build beat the 112.0 GiB flat rung above it — a
smaller artifact dominating a larger one, with a mechanism story ("the
low-bit lever") growing around it. It was false. The comparison table's
incumbent column held 2.8197, the score of a bf16-scales *proxy* build,
not the shipped artifact's real 2.7655 — a substitution our own notebook
had recorded and then forgotten [E79]. Re-scored against the real
artifact, the flat rung wins both corpora and the ladder is monotone; the
anomaly that motivated a day of mechanism work never existed. The number
was plausible precisely because it was real — it was just the score of a
different artifact. The gate that came out of it is procedural and
absolute: a comparison row must name the artifact AND the instrument that
produced it, and any number older than the artifact it faces is
re-measured, not cited [FINDINGS III.2]. The executable half,
`check_comparator.py`, exists because the failure has a silent twin: a
comparator that loads incompletely scores *worse* and flatters us
[FINDINGS III.3].

**5.2 The corrupt artifact that manufactured a 25x effect (→ outlier gate
before scoring).** The d2-versus-d4 geometry question was "settled" once
with a spectacular number: d4 better by 3.3x KL at matched size [E82]. The
d2 arm was a known-corrupt artifact — three tensors with relerr up to
0.988, produced by a machine whose outputs our own 08-15 record said must
be verified elsewhere before belief — and it had been scored without that
check [E85]. The tell was not the headline (a 3.3x win reads as a clean
landslide); it was a *monotonicity violation elsewhere in the sweep* — a
3.5 bpw rung scoring worse than a 3.0 bpw one, which more bits cannot do.
The clean re-run, both arms fit fresh on a trusted box and outlier-gated
first, gives the real effect: 12%, not 3.3x — the contaminated pair
overstated it ~25x [E87]. The rule is now structural: before scoring ANY
artifact, confirm it passed an outlier gate on a trusted box, because a
corrupt artifact scores plausibly and silently, and the fitter's own log
*cannot* see the corruption — it reports what it computed, not what
reached disk [FINDINGS III.9].

**5.3 The algebraic identity reported as a finding (→ cross-review).** The
mechanism-decider experiment [E91] produced two clean contrasts — fitter
effect −0.0109 at matched geometry, harvest cost +0.0042 at matched fitter
— and the resolution triumphantly noted that they "sum exactly" to the
measured total: additive, no interaction. A peer struck it within half an
hour: with three measurements sharing an endpoint, (a−c) = (a−b) + (b−c)
is an identity that closes on any data, including garbage; zero degrees of
freedom remained for an interaction to appear in [E91 correction]. The
honest phrasing — a decomposition *by construction*, each contrast
standing alone — is what this paper uses. Detecting a real interaction
would need the fourth cell of the 2x2, which nobody has fit.

**5.4 The units mismatch that impersonated a bias (→ pre/post-graft
stamping).** The size model's first two same-direction misses (−0.83,
−1.14 GiB) briefly looked like a flat-geometry bias in the model. The
resolution took a two-minute header read: all historic points were
measured post-graft and the new ones pre-graft, and the vision tower is a
fixed 912,020,960 bytes — measured directly, byte-identical across two
independently grafted artifacts, rather than recalled as "0.85 GiB" from
memory [night logs 08-21]. Corrected, one point became the best hit in the
series. Every size point is now stamped pre- or post-graft [law 5], and
the episode supplied this paper's epigraph.

**5.5 The gates, as a system.** Beyond the incident-driven rules:
pre-registration of predictions before fitting or scoring, with falsified
predictions recorded as falsified, never reframed [III.1; practiced at
E72→E74, E86→E87, E89's reading grid]; every new gate must fail on a
known-bad input AND pass on a known-good before it is trusted [III.5 —
the second half added after a gate false-alarmed on four healthy
comparators, E77]; packed bytes only, never stored bytes [III.8 — three
wrong conclusions in one day traced to this, E83/E87]; speed numbers with
n≥3, scatter, and stated prompt length [III.4]; scripts md5-checked against
HEAD in every chain preamble (`check_scripts_sync.sh`); bundled runtime
hash-compared against the runtime that produced the benchmarks
(`check_bundle`, E81/E90); placeholder tokens machine-checked before any
card upload (`check_card_placeholders.sh` — a human reader's eye supplies
meaning for `__TOKENS__`, so review alone cannot catch them); and laws
cited with commit hashes, because this project's own FINDINGS file moved
twice in 25 minutes one night and a peer built an argument on the stale
version [III.7].

None of this is novel methodology; it is pre-registration and unit
discipline under deadline conditions, applied by two people
cross-reviewing each other's numbers at 3am. What we can add from
experience is the negative space: across two weeks of dense measurement
(08-08 to 08-21, ~95 logged experiments), not one wrong number was caught by
someone looking at it and finding it implausible. The wrong numbers were the plausible ones.

## 6. Limitations — [STUB]

*To draft:* two MoE families (Qwen3.5-397B, Qwen3.6-35B), dense pending
`[[PENDING: E95 dense]]`; MLX/Metal single vendor stack; 35B prefill ~0.5x
affine even with u8view [E81]; no fast kernel for d8-class codebooks
(>32 KB threadgroup) [E83] — d8 quality verdict `[[PENDING: E89 d8
verdict]]` decides whether that kernel is worth building; fitter-vintage
effect real but mechanism unidentified [E94]; claim 2 measured on MoE only
(no dense harvest rung has been fit).

## 7. Reproducibility — [STUB]

*To draft:* HF artifacts under TheDrainFlorist, predecessor revisions
pinned and downloadable (comparison rows stay checkable) [card G]; bundled
model.py = benchmarked runtime (check_bundle); exo sharding requires
codebook replication — upstream PR #2268 stalled, fork
noahzelezny/exo:vq-codebook-replicate [card G]; all fit/pack/verify/score
scripts in-repo; referee corpora and their disjointness from any fitting
data (trivially: nothing was fit on data); upload checklist gates.

## Acknowledgments — [STUB]
Peer sessions' cross-review is load-bearing throughout §5; Dr. Saamer
Saab Jr. as first reader.
