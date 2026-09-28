# Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes Below 6 Bits

**Noah Zelezny**

*2026 · doi:[10.5281/zenodo.22119017](https://doi.org/10.5281/zenodo.22119017).
Every quality comparison is paired on identical token positions and read
against a measured fit-to-fit noise floor.*

## Abstract

Memory, not compute, is the binding constraint on local inference of
large language models: a model is runnable on a given machine only if
its weights fit in that machine's RAM, and for consumer and workstation
hardware — tens to a few hundred gigabytes of unified memory — this excludes, at full
precision, nearly every model approaching frontier performance. The gap is closed by quantization,
the family of techniques that store each weight in a few bits rather
than sixteen. The prevailing approach, affine quantization, rounds each
weight independently onto a uniform grid; its most refined variants tune
the grid per-channel on a calibration corpus, preserving more of the
original model per gigabyte of stored size. This paper evaluates an
alternative: **vector quantization (VQ)**, in which weights are grouped
into vectors of d consecutive values — d is the *dimension* — and each
vector is stored as an index into a *codebook*: a table of K
representative d-dimensional vectors learned by k-means over the weight
matrix itself.
Storage cost is log2(K)/d bits per weight, and the method is entirely
data-free: no calibration corpus, no activations, no teacher model.

We construct VQ quantizations of three models — Qwen3.5-397B-A17B and
Qwen3.6-35B-A3B (mixture-of-experts), and Qwen3.8-27B (dense) — and
compare them against both uniform and mixed-precision affine
builds of the same models at comparable file sizes, each size difference
stated. Every model
is scored on one deterministic instrument: KL divergence — how far the
quantized model's next-token probabilities drift from the full-precision
original's, reported in millinats (defined in §2.6; zero means identical
behavior) — paired over the same 12,288 token positions on each of three
corpora: prose, code and literary text.
Three findings. **First,
below approximately 5 bits per weight, the data-free VQ builds
outperform the affine builds.** On the 397B model, a 2.34-bit-per-weight
VQ build is 12.6 GiB smaller than the leading community build — a
mixed-precision artifact at 2.6 bits per weight — and diverges
less from the full-precision model on all three corpora (prose −30%, code
−10%, literary −44%); a 2.58-bit VQ build, 1.4 GiB smaller than the same
comparator, halves its prose divergence.
On the 35B model, a 3.68-bit VQ build is 3.3 GiB smaller than
the 4-bit affine build and diverges less on prose (−30%) and code (−34%),
with literary text a tie; a 4.37-bit VQ build, 0.5 GiB smaller than the same
comparator, halves its prose and literary divergence and cuts code by 44%. On the dense 27B, two
VQ builds straddle the 4-bit affine build's size: the one 0.5 GiB smaller
beats it on every corpus (prose −18%, code −9%, literary −8%), and the one
0.5 GiB larger by 10–34%. The advantage has a measured boundary: the affine
frontier overtakes VQ between about 4.5 and 6 bits per weight — bracketed at
4.7–5.6 on the dense model and 5.3–6.2 on the MoE — and at 8 bits affine
quantization is essentially lossless, leaving nothing to improve upon. VQ's regime is the low-bit range — which is precisely the
range in which large models fit on the hardware most people have.
**Second, model size becomes continuously tunable**: a two-coefficient
size model predicts an artifact's packed size to within a few tenths of
a GiB before it is fit — validated on all three models against builds
whose sizes were predicted before the builds existed — and
reducing the codebook size in the first layers alone reaches the sizes between
uniform geometries. **Third, weight-space
reconstruction error — the statistic most quantization pipelines
optimize and gate on — does not rank output quality in this regime.** A
fitter change that improves precisely the reconstruction statistic
identified as decisive, at identical bytes, degrades literary text by
35–60% on both MoE models while leaving code unchanged on one and improving it
10% on the other. Only evaluation of the assembled
model ranks artifacts.

Eleven VQ builds of the three models are published with pinned revisions,
and every comparison names the build and instrument that produced it.

## 1. Introduction

The feasibility of running a large language model locally is decided primarily by
memory. Compute per token is modest for models with few active
parameters — Qwen3.5-397B-A17B, the largest model studied here,
activates 17 of its 397 billion parameters per token — but the weights
must reside in RAM regardless of how many are active, and RAM is the
commodity in shortest supply outside the datacenter. At 16-bit
precision the 397B model needs about 740 GiB, beyond any consumer or
workstation machine. Quantization decides how low on the memory ladder a given model stays usable, and for the machines in
question the decisive band is often 2 to 6 bits per weight.

The available quantizations in this band are primarily affine: uniform
builds published by the mlx-community project, and mixed-precision
builds from the community at large, which raise attention and other
structure above the experts' width (for the 397B model we compare
against the most capable we could obtain, the spicyneuron 2.6-bit and
3.5-bit builds). The refined end of the affine family
tunes its grids on data — GPTQ orders quantization by approximate
second-order information from a calibration set [1], and AWQ scales
channels by activation statistics [2].

Vector quantization of LLM weights is itself established in the CUDA
ecosystem: GPTVQ interleaves Hessian-guided VQ with updates to the
remaining weights [3], AQLM learns additive multi-codebook
quantization on calibration data with end-to-end fine-tuning [4], and
QuIP# combines incoherence processing with E8 lattice codebooks [5].
All three are calibration-dependent, and none publishes runnable
artifacts for the Apple-Silicon MLX stack [8]; calibrated VQ builds for MLX,
with codebooks fit under GPTQ-style error compensation, have since been
published independently [9]. VPTQ [12] and QTIP [13] extend vector and
trellis-coded quantization in the same calibration-based setting.
Non-uniform scalar codebooks predate all of these: Deep Compression shares
k-means centroids within each layer and then retrains them [10], and
SqueezeLLM fits sensitivity-weighted k-means per channel on calibration
data [11]. HQQ is a data-free affine quantizer that optimizes each group's
scale and zero-point against the weights alone [14]. The lineage of the
method itself is older still: a codebook fit by k-means [6] over
fixed-length subvectors is the building block of product quantization [7].
Here one codebook is shared by every subvector of a tensor, it is fit to the
weights alone, and nothing is retrained.

**Notation.** A VQ geometry is written dN/KM, where N is the subvector
dimension and M the codebook size; builds in this paper range from
d2/K16 to d4/K16384. For example, d4/K2048 groups weights into
subvectors of 4 consecutive values and replaces each with an index into
a 2048-entry codebook, storing log2(2048)/4 = 2.75 bits per weight.
Every size in this paper is the measured size of the packed text weights
on disk; every quality number is measured on the assembled model.

The method is deliberately minimal: per-tensor k-means over the weight
subvectors, one flat codebook width across the whole surface, no data
anywhere in the loop. Data-free matters for more than elegance. A
calibration-fitted quantizer's quality is partly a property of its
calibration set, so any evaluation resembling that set flatters it and
any deployment unlike it is off-distribution. A weight-space fit has
nothing to inherit: it cannot be tuned toward an evaluation even by
accident, so its scores mean the same thing on any text. And because the
quantizer never sees data, every quality number comes from scoring the
assembled model, which §4 shows is the only reliable ranking.

**Claim 1 (method).** At equal or smaller packed size, the data-free
VQ builds beat the affine builds — mixed-precision and uniform — on all
three models (§3). The claim is bounded on both ends: the wins are
measured from 2.3 to 5.3 bits per weight, and the crossover where the
affine frontier passes above ours is bracketed at 4.7–5.6 bits on the
dense model and 5.3–6.2 on the 35B MoE, measured from both sides in
each case. At 8 bits affine is essentially lossless and there is
nothing to beat. The quality advantage also carries a cost: on the 35B,
VQ decode throughput is 0.77× and prefill 0.86× that of a same-size affine
build (§3.5).

**Claim 2 (size targeting).** A quantization can be tuned to a byte
budget. Codebook widths land where log2(K)/d puts them — on the 397B
each doubling of K at d4 adds about 11 GiB — and we make the axis
continuous: a two-coefficient size model prices any target before the
fit runs, and harvesting bits from the shallow layers, which tolerate
them, reaches the sizes between uniform geometries (§3.4).

**Claim 3 (measurement).** Weight-space reconstruction error is not a
reliable ranking of output quality, and cannot steer design. We show
this by construction, on two models: a fitter change that reduces
reconstruction error on the largest-magnitude weights, at identical bytes,
leaves the model
worse on prose and literary text on both models (+15%, +35–60%) while
code is unchanged on one and 10% better on the other (§4.3).


## 2. Method

The recipe has one moving part. In the mixture-of-experts models the
routed experts hold most of the parameters — 97% in the 397B and 93% in
the 35B; in the dense
model the same role is played by the MLP trio — the three feed-forward
projection matrices in each transformer layer (gate, up, and down),
which together dominate a dense model's parameter count. These
byte-dominant surfaces are the quantization target. A fixed non-expert skeleton is quantized
affinely once and never varied, and the target tensors are replaced by a
vector quantization whose dial is the geometry (d, K), held flat across
every such tensor. A build is named by its geometry.

### 2.1 The skeleton

All 397B and 35B builds share one base, chosen by measurement before any
VQ build and then held fixed: 6-bit structure, the linear-attention input
projections at 4-bit, and the routers at bf16. The routers are small (40 MiB
on the 35B, 240 MiB on the 397B), and a router feeds an argmax over experts,
where reduced precision changes which expert runs. Because the skeleton never varies, every VQ build of a
model differs from its siblings only in the expert tensors. The dense 27B builds splice VQ MLPs into a 4-bit
affine conversion, carrying every other tensor through unchanged, which
makes each build a controlled ablation of the MLP treatment against its
base. The vision tower is kept at bf16 in every build and is excluded
from every size in this paper, as is the optional MTP draft head: sizes
are text weights, the bytes mlx-lm loads.

### 2.2 The fit

For each target tensor independently: reshape the weights into
d-dimensional subvectors, scale each group of 64 by its maximum absolute
value, fit a K-entry codebook by k-means (k-means++ initialization, plain
Lloyd iterations), and store codes, scales and codebook. Two properties
matter downstream. Healthy reconstruction error scales with K — a fit at
K=128 sits near 0.46 relative error and a healthy K=2048 fit near 0.19 — so
acceptance thresholds are set per geometry. And the initialization
subsamples the weights stochastically, so two fits of the same tensor
differ; §2.6 measures the consequences and every comparison in this paper is
read against them.

Every build made for this paper uses exactly this fitter. The published
artifacts were fit with earlier versions of it and keep their original
codebooks:

| artifact | initialization | notes |
|---|---|---|
| 397B VQ-2.4bpw | random | first-generation fitter (frozen as `research/archive/quantlab/fitter_0816_cdcdeab.py`) |
| 397B VQ-2.6bpw, VQ-3.1bpw | k-means++ (the default from 2026-08-18) | |
| 397B, all three artifacts, layers 57–59 | k-means++ | 9 of 180 expert modules refit in 2026-09 with scale–codebook alternation (§3.2) |
| 35B VQ-3.8bpw | k-means++ | |
| 35B VQ-3.4bpw, VQ-4.6bpw, VQ-5.4bpw | not recorded | |
| 27B VQ-3.9bpw, VQ-4.5bpw | k-means++ | unseeded |
| 27B VQ-4.8bpw | k-means++ | seeded |

All use max-abs scales per group of 64. Differences between fitter versions
are bounded, together with draw-to-draw spread, by the floors of §2.6, which
compare a fresh fit with the current fitter against a published original.

### 2.3 Packing

Codes are packed to their true bit-width after fitting; packing is
bit-exact, verified at the logit level. Byte-aligned code widths are
stored directly (packing them saves nothing and costs decode speed).
All sizes are packed sizes measured on disk, and a row's size and its
quality always come from the same artifact.

### 2.4 Size targeting

Uniform geometries leave gaps between adjacent codebook sizes. To reach a
size inside a gap we *harvest bits*: hold the geometry of the deeper layers
fixed and reduce K in the shallow layers only (the first ten, at 397B scale),
which tolerate cheap bits. The
resulting size is predicted before the fit by a two-coefficient model —
at 397B, `new = base − 1.87 GiB × shallow_bits`; on the dense 27B and
the 35B, `total = code_bytes + scales + carry` with the carry measured
once per model. Out-of-sample records for both forms are in §3.4.

### 2.5 The pipeline

Every build passes, in order: fit → reconstruction-error check → pack →
merge into the base checkpoint → structural verification → generation of a
test output through the exact runtime the artifact ships with → scoring. The
generation step is necessary: scoring exercises a prefill-shaped code path
while serving exercises the fused decode kernels, and an artifact can score
normally while being unable to serve.

### 2.6 Instruments and noise floors

Three quantities appear throughout. **KL divergence**, reported in
millinats (mnats), measures how far the quantized model's next-token
probability distribution drifts from the full-precision model's, averaged
over a fixed token stream. A *nat* is the unit of information in the
natural logarithm, as a bit is in base 2; a millinat is a thousandth of
one. Zero means the quantized model behaves identically. **Top-1
agreement** is the fraction of positions at which the quantized model's
most probable token matches the full-precision model's. **Relative
reconstruction error (relerr)** is a weight-space quantity — the norm of
the difference between a tensor and its quantized reconstruction, relative
to the tensor's norm — used only as a corruption check, because §4.3 shows
it does not rank output quality.

**One instrument for every model.** Each unquantized bf16 model — the
*teacher* — is run once over three fixed corpora (prose, public code and
literary text), 12,288 tokens each in chunks of 512, and its full
log-probability distribution over the 248,320-token vocabulary is cached at
every position. The 397B teacher does not fit in memory on any machine we
have, so it is streamed layer by layer from disk for this one pass. Every
quantized build is then scored against those caches: exact KL over the full
vocabulary, and top-1 agreement, at each of the same 36,864 positions.
Because every build of a model sees the same positions against the same
cached teacher, two builds are compared *paired*: a per-position difference,
reported with its paired t statistic. We call a difference significant at
|t| > 2. Positions within a chunk are correlated, so this t overstates
confidence; the noise floors below are the stricter reference, and a margin
is claimed only where it also clearly exceeds the floor for its model. Scoring is deterministic
— an artifact reproduces its KL to every printed digit — and one scoring
path and batching is used throughout, because some architectures shift
log-likelihoods measurably with batch composition.

**Three corpora, reported separately.** Every model is scored on the
same three fixed texts, which ship with the code (§7). *Prose* is the first
12,288 tokens of a WikiText-2 excerpt of Wikipedia articles. *Code* is six
source files from the MLX framework at v0.30.0, in Python, Metal and C++.
*Literary* is the opening chapters of Jane Austen's *Pride and Prejudice*:
the corpus file holds excerpts from ten public-domain works, but only its
first 12,288 tokens are scored, and all of them fall within that novel. We report
the three separately rather than pooling them into one mean, because
quantization damage is not uniform across kinds of text: the same pair of
builds can rank one way on prose and the other on literary text (§3.3), and
a pooled mean is dominated by whichever corpus diverges most. All three
texts are public and plausibly present in the models' training data. That
shifts the level of every KL; it does not bias the comparisons, which are
paired on identical positions.

We do not rank by perplexity. It measures a model against the text rather
than against the model it approximates, and it aggregates away offsetting
errors: on the 27B it ranks the affine builds in the wrong order (§3.3).

**Noise floors.** Two fits of identical geometry differ, because k-means
initialization draws a random subsample. We measure that spread with a
second, independent fit of the same geometry, made with the current fitter
and a different random initialization and scored paired against the
original. One such pair per model lands within 0.1–5.8% on every corpus,
and no difference exceeds |t| = 2.5:

| model | geometry | prose | code | literary |
|---|---|---|---|---|
| 27B | d2/K256 | +0.1% (t +0.0) | +2.3% (+0.8) | +3.4% (+2.5) |
| 35B | d2/K1024 | +0.7% (+0.4) | +0.2% (+0.1) | −5.8% (−1.0) |
| 397B | d4/K128 | +1.4% (+0.9) | +5.2% (+2.4) | −2.7% (−1.0) |

*Original minus second fit, full-vocabulary KL, 12,288 paired positions per
corpus. The originals keep the fitter version they were built with (§2.2),
so each floor bounds draw-to-draw and fitter-version spread together.*

Each floor is a single pair of fits at one geometry per model; §3 reads every
margin against the floor for its model. Two consequences: a |t| near the
floors' own maximum of 2.5 is not treated as a difference, and a margin is
claimed only where it is well outside the few-percent spread above.

The fitter is seeded by default (seed 1234): a fit is reproducible
bit-for-bit from its recipe and seed. Most artifacts in this paper predate
that default and are single unseeded draws. Seeding makes an artifact
reproducible; it does not remove the draw-to-draw spread, which is why the
floors exist and why no margin is read without one.

## 3. Results

### 3.1 Geometry: what d and K buy

Rate is log2(K)/d, so the same bit rate is reachable with small vectors
and small codebooks or large vectors and large codebooks. Measured at
matched code rate, builds of byte-identical size (d4 minus d2, negative =
d4 better):

| code rate | pair | prose | code | literary |
|---|---|---|---|---|
| 2.00 bpw (35B) | d4/K256 vs d2/K16 | −14.6% (t −10.2) | −16.7% (−19.7) | −13.0% (−8.1) |
| 3.00 bpw (27B) | d4/K4096 vs d2/K64 | −3.5% (−2.2) | −14.2% (−5.1) | −10.7% (−9.5) |

Dimension pays at matched rate on every corpus of both models, by 3–17%. It also has costs. d4 has a hard
rate ceiling of 4.0 bpw (16-bit indices over 4 weights, even at a
65,536-entry codebook), so the high bands belong to d2. And large
codebooks outgrow the GPU's fast on-chip memory: Apple's threadgroup
limit is 32 KB, which a d4 codebook exceeds above K = 2048 and a d2
codebook above K = 4096, and larger codebooks are served from device
memory instead. The operational sweet spots this induces: d4 with the
largest codebook that fits the band, d2 above 4 bpw.

### 3.2 The 397B ladder

![397B ladder](fig_397b_ladder.png)

Sizes are text weights: the safetensors mlx-lm loads, excluding the
0.85 GiB bf16 vision tower and the optional 5.4 GiB MTP draft head that
ship beside them. That is the like-for-like basis against the community
comparator, which carries neither. KL is paired over the same 12,288
positions per corpus against the bf16 teacher (§2.6).

**Ours (VQ, uniform d4 across all 60 layers; KL in mnats):**

| build | release | GiB | bpw | prose | code | literary | prose top-1 |
|---|---|---|---|---|---|---|---|
| flat d4/K128 | — | 96.7 | 2.10 | 354.4 | 127.8 | 275.0 | 82.2% |
| **flat d4/K256** | **VQ-2.4bpw** | 108.0 | 2.34 | 232.7 | 89.0 | 134.1 | 86.4% |
| **flat d4/K512** | **VQ-2.6bpw** | 119.2 | 2.58 | 166.1 | 57.8 | 60.4 | 88.5% |
| **flat d4/K2048** | **VQ-3.1bpw** | 141.7 | 3.07 | 93.0 | 33.3 | 16.6 | 91.7% |

Bold rows are published artifacts, under
`TheDrainFlorist/Qwen3.5-397B-A17B-<release>`. The d4/K128 build is a ladder
point only, not a published artifact. The bpw column is measured: text bytes over the model's
396.35 billion text parameters. Release names identify artifacts and were
set at release; VQ-2.4bpw measures 2.34 because the correction below
removed 2.8 GiB from it after it was named. Every bpw figure in the text is
this measured one, which counts the codes, their scales and the affine
skeleton together. The codebook rate alone is lower: d4/K2048 stores
11-bit indices, 2.75 bits per quantized weight before scales.

**Community affine (mixed precision by component, text-only):**

| build | GiB | bpw | prose | code | literary | prose top-1 |
|---|---|---|---|---|---|---|
| spicyneuron 2.6bit | 120.6 | 2.61 | 333.9 | 99.2 | 241.1 | 83.5% |
| spicyneuron 3.5bit | 165.6 | 3.59 | 87.7 | 31.0 | 17.0 | 91.8% |

Both comparators split by component: attention, the shared experts, the
embeddings and the output head at 8-bit, the linear-attention input
projections at 4-bit, and the routed experts at the default width — 2-bit
in the 2.6-bit build, 3-bit in the 3.5-bit build — with 377 promoted
modules across all 60 layers in each. They are complete, fair opponents.

Two comparisons carry claim 1 here, and neither needs matched bytes,
because the VQ build is smaller and better at once. **d4/K256 against the
2.6-bit build:** 12.6 GiB smaller, with less divergence on every corpus —
prose −30% (paired t = −17.1), code −10% (t = −3.8), literary −44%
(t = −14.2). **d4/K512 against the same build:** 1.4 GiB smaller, prose
−50% (t = −28.2), code −42% (t = −14.0), literary −75% (t = −25.5). Each t
is paired over the same 12,288 positions (§2.6). Top-1 agreement moves with
KL on every corpus; there is no inversion.

At the top of the ladder the result is parity rather than dominance.
**d4/K2048 against the 3.5-bit build:** 23.9 GiB smaller; indistinguishable
on code (+7.3%, t = +0.9) and literary text (−2.2%, t = −0.2), and 6.1% worse
on prose (t = +2.1, within the range the noise floors reach), with top-1 agreement 91.7% against
91.8%. The claim there is near-parity at 14% fewer bytes.

The ladder also locates the crossover against the 2.6-bit build. d4/K128,
23.9 GiB smaller than it, is worse on all three corpora: prose +6%
(t = +3.3), code +29% (t = +9.2), literary +14% (t = +4.0). An independent
second fit of d4/K128 lies within 1.4%, 5.2% and 2.7% of the first on the
three corpora (§2.6), so each deficit exceeds the draw-to-draw spread. Uniform VQ overtakes this
affine build between 96.7 and 108.0 GiB, just under 2.4 bits per weight.

**Correction to v4.** The v4 rows for d4/K128, d4/K256, d4/K512 and
d4/K2048 were measured on builds whose expert modules in layers 57–59 (9 of
180) were affine 3-bit rather than VQ. All builds above are uniform across
layers 0–59; the replacement fits for those 9 modules were made with the
scale-alternating fitter variant (§2.2). The correction changed prose KL by
+0.28 (not significant), +2.32, +4.94 and +12.96 mnats at d4/K2048, K512,
K256 and K128 on the top-64 KL instrument used in v4. The v4 row labeled
flat d8/K16384 (VQ-2.2bpw) was a mixed-geometry build and is withdrawn from
this uniform ladder.

### 3.3 The 35B MoE and the dense 27B

![35B and 27B ladders](fig_35b_27b.png)

Sizes, bpw and KL are on the same basis as §3.2: text weights, measured bits
per weight over 34.66B (35B) and 26.90B (27B) text parameters, and
full-vocabulary KL in mnats over 12,288 paired positions per corpus.

**35B — ours (VQ, uniform geometry across all 40 layers):**

| build | release | GiB | bpw | prose | code | literary | prose top-1 |
|---|---|---|---|---|---|---|---|
| d2/K16 | — | 10.14 | 2.51 | 231.9 | 1283.8 | 1095.8 | 81.0% |
| **d4/K2048** | **VQ-3.4bpw** | 12.96 | 3.21 | 76.6 | 529.1 | 208.0 | 89.4% |
| **d4/K8192** | **VQ-3.8bpw** | 14.84 | 3.68 | 46.1 | 430.8 | 101.9 | 91.4% |
| d4/K16384 | — | 15.78 | 3.91 | 39.8 | 393.1 | 80.5 | 92.2% |
| d2/K256 | — | 17.64 | 4.37 | 31.0 | 361.8 | 52.6 | 93.2% |
| **d2/K1024** | **VQ-5.4bpw** | 21.39 | 5.30 | 22.6 | 349.6 | 33.7 | 94.0% |
| d2/K4096 | — | 25.15 | 6.23 | 20.8 | 336.4 | 31.7 | 94.2% |

Bold rows are published artifacts, under
`TheDrainFlorist/Qwen3.6-35B-A3B-<release>`. One further published 35B build,
VQ-4.6bpw, mixes two geometries (d4/K2048 in the first 10 layers, d2/K512
after) and is not a point on this uniform ladder.

**35B — affine:**

| build | GiB | bpw | prose | code | literary | prose top-1 |
|---|---|---|---|---|---|---|
| q3 | 14.14 | 3.50 | 259.8 | 1018.3 | 868.4 | 77.6% |
| q4 | 18.17 | 4.50 | 65.4 | 647.8 | 105.6 | 88.9% |
| q6 | 26.23 | 6.50 | 9.5 | 244.4 | 16.0 | 96.1% |
| q8 | 34.30 | 8.50 | 5.7 | 175.5 | 10.8 | 97.0% |

The q4 and q8 rows are configuration-identical to the mlx-community
conversions of this model; q3 and q6 are ours, by the same converter.

One asymmetry in these 35B comparisons runs in our favor, and it is not
visible in the sizes. Every affine comparator here quantizes the MoE
router — the community 4-bit and 8-bit and our own 6-bit all carry
quantized gate projections — while our VQ builds leave the routers at
bf16. The bytes are trivial, 40 MiB or 0.26% of the build. But a
router's output feeds an argmax over experts, so quantizing it can change
*which* expert runs rather than perturbing an output proportionally, and
the effect is therefore not bounded by the byte share the way an ordinary
precision difference would be. We have not measured it: that would take a
VQ build with routers forced to 8 bits, re-scored, and we did not run one.
The 397B comparisons in §3.2 are unaffected — there our builds and both
comparators keep routers at bf16.

Code KL is an order of magnitude higher on the 35B than on the other two
models, for every quantization including q8. The teacher cache reproduces a
direct forward pass exactly, and the mean is dominated by a heavy tail of
positions at which every quantized build routes to different experts than the
teacher. The orderings reported here hold for the mean, a trimmed mean and
the median alike.

At the small end VQ dominates. VQ-3.4bpw against q3: 1.2 GiB smaller, and
prose −70% (t = −53.6), code −48% (t = −26.7), literary −76% (t = −43.1).
VQ-3.8bpw against q4: 3.3 GiB smaller, with prose −30% (t = −17.0) and code
−34% (t = −16.0), and a tie on literary text (−3.5%, t = −0.6). d2/K256
against q4: 0.5 GiB smaller, and prose −53% (t = −32.5), code −44%
(t = −18.1), literary −50% (t = −10.8).

VQ-5.4bpw lands between q4 and q6 in size. Against the affine frontier
log-interpolated to its 21.39 GiB it is below the line on all three corpora
(22.6 against 30.3 prose, 349.6 against 438.6 code, 33.7 against 49.7
literary), and its independent second fit scores within 0.7%, 0.2% and 5.8%
of it (§2.6). One geometry higher the sign flips: d2/K4096 is 1.1 GiB smaller
than q6 and worse on every corpus — prose 2.2× (t = +28.6), code +38%
(t = +7.3), literary 2.0× (t = +7.2). (That two "6-bit" artifacts differ by
1.1 GiB is expected: a nominal rate names the code width on the quantized
surface, while total bytes include each method's scale overhead and its
treatment of the non-expert remainder — which is why every comparison in this
paper is by measured size, never by nominal rate.) **The crossover on this
model sits between 5.3 and 6.2 bits per weight.**

**27B — ours (VQ):**

| build | release | GiB | bpw | prose | code | literary | prose top-1 |
|---|---|---|---|---|---|---|---|
| d4/K256 | — | 9.61 | 3.07 | 372.8 | 135.4 | 1301.9 | 77.3% |
| d4/K1024 | — | 10.61 | 3.39 | 204.2 | 68.8 | 879.5 | 83.0% |
| **d4/K4096** | **VQ-3.9bpw** | 11.61 | 3.71 | 146.6 | 41.2 | 584.3 | 86.0% |
| **d2/K256** | **VQ-4.5bpw** | 13.60 | 4.34 | 56.3 | 21.5 | 289.2 | 90.9% |
| **d2/K512** | **VQ-4.8bpw** | 14.59 | 4.66 | 61.8 | 16.9 | 208.1 | 91.1% |
| d2/K4096 | — | 17.58 | 5.61 | 38.6 | 13.5 | 145.9 | 92.6% |

Bold rows are published artifacts, under
`TheDrainFlorist/Qwen3.8-27B-<release>`.

**27B — affine.** Unlike the 397B and 35B comparators, these builds are our
own conversions.

| build | GiB | bpw | prose | code | literary | prose top-1 |
|---|---|---|---|---|---|---|
| q2 | 7.83 | 2.50 | 1341.6 | 995.6 | 2649.0 | 51.2% |
| q3 | 10.96 | 3.50 | 192.4 | 86.3 | 965.1 | 82.3% |
| q4 | 14.09 | 4.50 | 68.5 | 23.7 | 315.6 | 90.5% |
| q6 | 20.36 | 6.50 | 10.9 | 2.2 | 24.7 | 96.9% |
| q8 | 26.62 | 8.50 | 3.5 | 0.6 | 5.0 | 98.5% |

The recipe is not an MoE phenomenon. VQ-3.9bpw against q3, 0.65 GiB larger:
prose −24% (t = −12.7), code −52% (t = −22.2), literary −39% (t = −41.9).
d4/K1024, 0.35 GiB smaller than q3, beats it on code (−20%, t = −8.4) and
literary (−9%, t = −9.1) and is 6% worse on prose (t = +2.8). VQ-4.5bpw
against q4, 0.5 GiB smaller: prose −18% (t = −6.5), code −9% (t = −3.3),
literary −8% (t = −6.3); VQ-4.8bpw, 0.5 GiB larger: prose −10% (t = −3.4),
code −29% (t = −10.0), literary −34% (t = −28.8). Above, the picture
inverts: q6, 2.8 GiB larger than d2/K4096, has 3.5× less prose KL, 6.0× less
code and 5.9× less literary; and d2/K4096 lies above the affine line
log-interpolated to its size on all three corpora. **The dense crossover is
bracketed at 4.7–5.6 bpw.**

One instrument note: on this instruction-tuned model, prose perplexity ranks
the affine builds wrongly. q4 scores 5.683, *below* q6 (5.965) and q8 (5.947),
while its KL is 6× that of q6 and 20× that of q8, and its top-1 agreement
6–8 points lower.
Perplexity cannot rank quantizations here; divergence and agreement can.

### 3.4 Size targeting

The size models' out-of-sample record, each prediction made before the build
existed: at 397B, six of seven predictions within 0.4 GiB (best 0.02 GiB)
and the seventh within its stated band; on the 35B, three geometry
predictions at −0.03%, −0.30% and −0.37%; on the dense 27B, three builds
across two geometries within 0.003 GiB.

Bit harvesting (§2.4) makes size continuously variable between uniform geometries: holding the
body geometry fixed and reducing K in the shallow layers sheds a predictable
number of bytes. It is a sizing mechanism. Its effect on quality depends on
which layers keep the larger codebooks, a question of mixed-geometry
allocation that is outside the scope of this paper; no quality claim here
rests on a bit-harvested build. Together with the size model, the capability is:
name a byte budget, price the build, fit it once.

### 3.5 Runtime performance and kernel support

None of this serves without custom Metal kernels: a fused
decode-and-matmul path that reads codes and codebook directly (per-K
bit-width extraction in-kernel), a device-memory codebook variant for
the codebooks that exceed Apple's 32 KB threadgroup memory, and a
zero-copy view that dispatches byte-aligned unpacked codes through the
packed kernel (+33% prefill, bit-exact). All kernel variants are
accepted only on bit-identity with a reference path where both load,
and on relative error against a float32 reference where only one does.

Speed against affine, measured on the 35B: the d2/K256 build (17.64 GiB)
against q4 (18.17 GiB), a 2,048-token prompt from the prose corpus and 128
generated tokens, one fresh process per arm per run, arms alternating, three
runs each, in one session on an otherwise idle machine:

| | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| decode, VQ / affine | 0.73 | 0.78 | 0.77 | 0.77 |
| prefill, VQ / affine | 0.78 | 0.86 | 0.88 | 0.86 |

*Peak memory 20.6 GB (VQ) and 21.4 GB (affine). Apple M3 Ultra.*

VQ costs about a quarter of decode throughput and a seventh of prefill
throughput at this prompt length, against a build with more than twice the divergence on
prose (§3.3). Shorter prompts were not measured at n ≥ 3 and are not
characterized here. Speed numbers are same-session ratios between arms: we
found decode throughput at ~100 GiB residency to be bimodal on our hardware
(the same build varying 40% run to run), and we therefore publish no
absolute throughput figures. Three further kernel changes — fused
row-gather, byte-aligned packing and native-bf16 kernels — each gave no
speedup or a slowdown.

## 4. Negative results

This section reports what did not work and what cannot be reached. Each
boundary was measured, and several bound the claims of §3.

### 4.1 The 8-bit ceiling

At 8 bits affine is essentially lossless: prose KL 5.7 mnats on the 35B
(34.30 GiB) and 3.5 on the 27B (26.62 GiB), both uniform 8-bit conversions
with no per-module overrides (§3.3).

Nothing below 8 bits approaches that, for either method. On the 35B, q6
(26.23 GiB) has 1.7× the prose KL of q8 and VQ-5.4bpw (21.39 GiB) 4.0×; on
the 27B, q6 (20.36 GiB) has 3.1× and d2/K4096 (17.58 GiB) 11×. On both
models, 8-bit quality costs 8-bit bytes, for affine and for VQ alike.

### 4.2 Where the geometry axes stop paying

Dimension pays at matched rate (§3.1): d4 over d2 by 13–17% at 2.0 bpw and
3–14% at 3.0, depending on the corpus.
Whether it still pays at d4's 4.0 bpw ceiling is untested. The only d4
geometry that reaches that rate uses a 65,536-entry codebook: a single fit
of it on the 27B takes about 37 hours, its 512 KB codebook is sixteen
times the 32 KB threadgroup limit and so could never use the fast decode
path, and codebook size is already in steep diminishing returns well below
it (next). It was not run, and the dimension advantage is not established
above 3 bpw. Codebook size pays with steep
diminishing returns: on the 35B, flat d4 at K2048, K8192 and K16384
scores 76.6, 46.1 and 39.8 mnats of prose KL — quadrupling K removes 40%
of the divergence, and doubling it again removes a further 14% for 0.9 GiB.

### 4.3 Reconstruction error does not rank output quality

A refit of the published 397B VQ-2.4bpw at its own geometry (d4/K256,
byte-identical size) has *lower* mean reconstruction error on every
projection, yet it is no better as a model: +3.9% KL on prose (t = +2.0),
−4.7% on code (t = −1.9), −4.8% on literary (t = −1.1). Percentile analysis
of the weights locates the trade: the refit is better where most weights live
and worse in the top 0.1% by magnitude, and mean reconstruction error, a bulk
statistic, reports the trade as an improvement. The pattern holds across 36
tensors and has a direct cause: body-layer weights are sub-Gaussian, so a
codebook with lower average distortion buys it from the tail.

Reweighting the k-means objective toward that tail band (weights ∝ |w|^4,
body layers only) does what it is built to do in weight space. On one 35B
module it cuts relative error on the top 0.1% of weights from 0.192 to 0.110
and on the top 1% from 0.202 to 0.130, at a mean-error cost of 0.313 → 0.374.
Its effect on the model depends on the model and on the text. Each pair below
is built twice with the same fitter, seed and byte-identical size, differing
only in this weighting:

| model | corpus | unweighted | tail-weighted | Δ | t |
|---|---|---|---|---|---|
| 397B, d4/K256 | prose | 241.8 | 279.1 | +15.5% | +7.0 |
| | code | 84.7 | 84.3 | −0.5% | −0.1 |
| | literary | 127.6 | 171.8 | +34.7% | +6.6 |
| 35B, d4/K256 | prose | 198.1 | 227.7 | +14.9% | +8.1 |
| | code | 1070.1 | 961.5 | −10.1% | −11.8 |
| | literary | 953.8 | 1521.8 | +59.6% | +29.1 |

*Full-vocabulary KL (mnats, 12,288 paired positions per corpus), all expert
modules refit. Weighting on layers ≥ 20 of 60 (397B) and ≥ 13 of 40 (35B).*

The same weight-space change leaves 397B code unchanged, improves 35B code by
10%, and degrades literary text by 35–60% on both, all far outside noise. No
single weight-space statistic can order these models, because their order
depends on the model and the corpus.

These comparisons share base weights and differ only in the fitter, so they
are unaffected by when, or on what software stack, a build was fit. They are
also large against fit-to-fit variation: an independent second fit of the same
geometry lands within 0.1–5.8% of the first on every corpus (|t| ≤ 2.5; §2.6).

## 5. Measurement discipline

The tables follow four recording rules.

A margin is read against the measured fit-to-fit floor for its model
(§2.6); a margin inside the floor is noise regardless of direction.

A comparison row names the build and the instrument that produced it. Every
number in §3–4 comes from one instrument (full-vocabulary KL on the same
cached teacher), and a row's size and quality come from the same build, with
size as text bytes on disk.

Each automated check was validated on a known-bad and a known-good input
before its verdicts were used.

No build is treated as releasable until it has generated tokens through the
exact runtime it ships with.

A published artifact is identified by its pinned Hugging Face revision.
Builds made since September 2026 also carry a build record naming the code
commit, inputs, fitter recipe, seed and a hash of every output file; the
artifacts in this paper predate it. Stored metadata is treated as a record of intent
rather than of content. A checkpoint's `model_type` field, for
example, names the loader code path rather than the model: Qwen 3.6
and 3.8 share the 3.5 architecture, so their checkpoints all declare a
`qwen3_5` variant, and every derived build inherits the stamp. A field
like that cannot identify what a file contains, so any stored property
that carries a claim is verified against the bytes themselves — a tensor compared with its claimed base, a hash
recomputed, a flag traced to the code that reads it.

## 6. Limitations

**Coverage.** Three models from one vendor family, only one of them
dense; a second dense model would test whether the 27B generalizes.
(The gemma-4 family showed similar behavior but cannot be scored
deterministically and is excluded.) Single software stack (MLX/Metal)
and one hardware platform; the kernel conclusions, including threadgroup
capacity, are specific to Apple Silicon. Quality is measured by KL to the
teacher on three corpora, not by downstream task benchmarks.

**Unmeasured regions.** The VQ/affine crossover is bracketed on the
35B and the 27B, but not on the 397B: affine builds above 3.5 bits
exist or could be produced for that model, but at ~225 GB for a 4-bit
build and ~320 GB for 6-bit they exceed the memory of any machine
available to this project, so whether the same crossover band holds at
that scale is untested. Whether dimension still pays at d4's 4.0 bpw
ceiling is also untested, for the cost reasons given in §4.2. Bit harvesting
has been built only on the MoE models.

**Instrument limits.** Each model's noise floor is a single pair of fits at
one geometry (§2.6), so the floor itself has no variance estimate, and the
paired t statistic overstates confidence because positions within a chunk
are correlated. Perplexity cannot rank quantizations on the
instruction-tuned 27B (§3.3). Decode throughput was bimodal at ~100 GiB
residency on our hardware and is uncharacterized at other sizes; we publish
ratios within a session, never absolutes. Speed was measured on one pair of
builds at one prompt length (§3.5).

**Costs.** On the 35B, VQ decode runs at 0.77× and prefill at 0.86× the
throughput of a same-size affine build (§3.5).

## 7. Reproducibility

All artifacts are published under `TheDrainFlorist` on Hugging Face
with their VQ runtimes bundled in-checkpoint: they load in stock mlx-lm,
unpatched, with `trust_remote_code=True`. Where a repository's weights were upgraded in place, the
previous build remains fetchable at its pinned revision and the card
labels which weights produced which benchmark rows. Most published fits are
unseeded single draws (§2.2): reproducible in recipe and geometry, not
bit-for-bit. Every build made for this paper is seeded.

Every VQ score in this paper was produced with the v2 runtime profile, in
which the VQ kernels take and return bf16. Published artifacts carry either
this profile or the earlier v1.5 profile, which differs only in those two
input/output settings; on the rows rescored both ways, the two profiles
agree within 2% KL. Each scored copy generated tokens through its bundled
runtime before scoring. Fit, pack, verification and scoring tools are published in the
project repository, **VQLab**
([github.com/noahzelezny/VQLab](https://github.com/noahzelezny/VQLab),
Apache-2.0). All three evaluation corpora ship with it: a WikiText-2 prose excerpt,
a public code corpus of six MLX source files, and a literary corpus of
excerpts from ten public-domain works (of which the scored span is the
first), each with a record of its source, license and exact
contents. Nothing was fit on
data, so there is no train/eval overlap to disclose.

## References

[1] E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers.*
ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).

[2] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao,
X. Dang, C. Gan, S. Han. *AWQ: Activation-aware Weight Quantization for
On-Device LLM Compression and Acceleration.* MLSys 2024.
[arXiv:2306.00978](https://arxiv.org/abs/2306.00978).

[3] M. van Baalen, A. Kuzmin, I. Koryakovskiy, M. Nagel, P. Couperus,
C. Bastoul, E. Mahurin, T. Blankevoort, P. Whatmough. *GPTVQ: The
Blessing of Dimensionality for LLM Quantization.* 2024.
[arXiv:2402.15319](https://arxiv.org/abs/2402.15319).

[4] V. Egiazarian, A. Panferov, D. Kuznedelev, E. Frantar, A. Babenko,
D. Alistarh. *Extreme Compression of Large Language Models via Additive
Quantization.* ICML 2024.
[arXiv:2401.06118](https://arxiv.org/abs/2401.06118).

[5] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even
Better LLM Quantization with Hadamard Incoherence and Lattice
Codebooks.* ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396).

[6] S. Lloyd. *Least Squares Quantization in PCM.* IEEE Transactions on
Information Theory 28(2):129–137, 1982.

[7] H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest
Neighbor Search.* IEEE Transactions on Pattern Analysis and Machine
Intelligence 33(1):117–128, 2011.

[8] A. Hannun, J. Digani, A. Katharopoulos, R. Collobert. *MLX:
Efficient and Flexible Machine Learning on Apple Silicon.* Software, 2023.
[github.com/ml-explore/mlx](https://github.com/ml-explore/mlx).

[9] aquaman164. *Qwen3.6-35B-A3B-MLX-VQ* (2.4, 2.6 and 3.4 bpw builds).
Software, Hugging Face, 2026.
[huggingface.co/aquaman164](https://huggingface.co/aquaman164).

[10] S. Han, H. Mao, W. J. Dally. *Deep Compression: Compressing Deep
Neural Networks with Pruning, Trained Quantization and Huffman Coding.*
ICLR 2016. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149).

[11] S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W. Mahoney,
K. Keutzer. *SqueezeLLM: Dense-and-Sparse Quantization.* ICML 2024.
[arXiv:2306.07629](https://arxiv.org/abs/2306.07629).

[12] Y. Liu, J. Wen, Y. Wang, S. Ye, L. L. Zhang, T. Cao, C. Li, M. Yang.
*VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language
Models.* EMNLP 2024. [arXiv:2409.17066](https://arxiv.org/abs/2409.17066).

[13] A. Tseng, Q. Sun, D. Hou, C. De Sa. *QTIP: Quantization with Trellises
and Incoherence Processing.* NeurIPS 2024.
[arXiv:2406.11235](https://arxiv.org/abs/2406.11235).

[14] H. Badri, A. Shaji. *Half-Quadratic Quantization of Large Machine
Learning Models.* Technical report, 2023.
[mobiusml.github.io/hqq_blog](https://mobiusml.github.io/hqq_blog/).

## Acknowledgments

The spicyneuron and mlx-community builds served as comparators
throughout; this work exists because those artifacts were public,
pinned, and worth measuring against. We hope ours are the same.

**AI disclosure.** The experiments in this work were executed with
substantial assistance from large language model agents (Anthropic
Claude, Opus and Sonnet-class models), which operated the fitting,
packing, verification, and scoring pipelines under the author's
direction, and assisted in drafting this manuscript. All quantitative
results were produced by the deterministic instruments described in
§2.6, are traceable to a committed laboratory record, and were verified
independently of any model-generated summary. The author directed all
experiments, made all methodological decisions, and takes sole
responsibility for the content.
