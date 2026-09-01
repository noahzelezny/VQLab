# Direct codebook sensitivity allocation — method note

**Status: developing. One family (GLM-5.3-Flash), one seed. Nothing here is
established until it replicates.** Named 2026-08-31 (Noah): DIRECT CODEBOOK
SENSITIVITY ALLOCATION — "direct" because it measures the optimized quantity
rather than a proxy. Name to be workshopped once findings exist; note that if
§4.1's affine test succeeds the method is not codebook-specific and the name
will want widening.

A method for deciding which layers of a VQ-quantized model get which codebook
geometry, under a byte budget. It differs from the mixed-precision allocation
literature in one respect that turns out to matter: it **measures the quantity
being optimized** rather than a proxy for it.

---

## 1. The method

Three steps.

**Sweep.** For each expert layer, swap that layer's codes to a different
geometry and score the model. Everything else is held fixed. Repeat per layer,
per available geometry.

**Solve.** The per-layer deltas are the coefficients of a multiple-choice
knapsack: each layer picks one geometry, each geometry costs or frees bytes,
maximize total gain subject to the budget. 42 layers x 3-4 levels solves
exactly by DP in milliseconds.

**Build and verify.** Splice the solution, gate it, and rescore. The solve is
additive and the surface is not, so the prediction is an upper bound and the
artifact is the answer.

### Why it is affordable

The enabling trick: **codebooks are swappable after fitting**. Two artifacts
fitted at different geometries share an identical `weight_map`, and their
per-module config entries differ only in `k`/`dim`/`pack_bits`. So a variant
costs one shard rewrite plus one score — NOT a refit. Measured on GLM-5.3:
~1.2-1.5 min per measurement against ~50 min for the cheapest full fit.

That is what makes exhaustive per-layer measurement practical, and it is the
part most worth stating plainly if this is ever written up: the method is not
clever, it is just cheap enough to brute-force.

### What it costs

Per rung: 42 layers x (levels available) measurements. At GLM-5.3 scale that
is ~1-2 h per rung. Splice dominates, so reducing scoring tokens helps less
than expected (512-token scoring cut per-sample time from ~1.5 to ~1.4 min,
not 4x).

---

## 2. Findings so far (GLM-5.3-Flash, seed 1234)

All measured; see research/glm53-flash/LEDGER.md for the raw numbers.

### 2.1 Layer identity dominates

At a fixed +3.38 GiB budget on the 98.55 GiB rung, the same bytes bought:

| selection | d_KL |
|---|---|
| hidden-state leverage probe, top 8 | +15.23 |
| low-leverage control, 8 | +16.89 |
| **measured effect, best 8** | **+54.98** |

3.6x between the best and worst 8-layer selection at identical size.

### 2.2 Measurement beats proxy, and the proxy ANTI-SELECTED

Against uniformly buying more bits everywhere (8.42 mnats/GiB on that rung):

| allocation | mnats/GiB | vs uniform |
|---|---|---|
| leverage-probe targeted | 4.51 | 0.54x |
| measured targeted | 16.27 | **1.93x** |

The leverage probe (per-layer local hidden-state damage, no compounding) put
three actively HARMFUL layers into its top eight. It is not merely
uninformative here; it selected worse than near-arbitrary choice. Its own
docstring called it a ranking instrument requiring referee confirmation — the
referee rejected it.

### 2.3 Allocation is NON-MONOTONIC

16 of 42 single-layer promotions made the model WORSE (worst: -5.23 mnats for
+0.42 GiB). Seven layers flip sign BETWEEN levels — hurting at K2048 but
helping at K8192. So "more bits in this layer" is not a monotone improvement,
and a layer can be worse at a middling precision than at either extreme.

Working explanation, untested: a layer reconstructed more finely falls out of
step with neighbours still coarse, breaking an error cancellation. This is a
hypothesis, not a result.

### 2.4 Sensitivity is BASE-SPECIFIC (the load-bearing negative result)

Rankings measured at one bit budget do not transfer to another:

| transfer | predicted | measured | note |
|---|---|---|---|
| 98.55-base table -> 116 rung, 34 layers | +121.71 | +21.20 | 0.17 realisation |
| 98.55-base table -> 134 rung, 9 layers | +11.99 | -15.36 | WRONG SIGN |
| **98.55-base -> 134 rung, ONE layer (L33)** | **+4.49** | **-6.87** | **WRONG SIGN, no interaction confound** |

The single-layer case is the proof: one change, no interactions, sign still
flips, and the 6.87 magnitude clears the 6.32 mnat seed floor. A layer's
response to precision depends on what precision the rest of the model carries.

**Consequence for the method:** sweeps must be run per rung. **Consequence for
the field, if it replicates:** allocation work that assumes a single global
per-layer sensitivity is measuring something budget-specific and calling it
intrinsic.

### 2.5 Interactions are real but small

Additive main effects explain R^2 = 0.678 of allocation outcomes across 40
random allocations, slope 0.710 (diminishing returns). Residual sd 4.59 mnats
— BELOW the 6.32 mnat seed floor. A surrogate model could fit that residual;
it would be optimizing beneath the noise of the codebook draw. Greedy on
measured main effects is within ~2 mnats of an exact DP solve.

Superadditivity is not a constant: weak sets beat their sum (1.74-1.82x),
strong sets fall short (0.84x), and mixed demote/promote sets fall further
(0.68x). It cannot be used as a fixed correction.

---

## 3. What would make this a paper

The method alone is an engineering note. The findings in 2.2-2.4 are the
contribution, and they are uncomfortable enough to be worth publishing IF they
survive:

1. **Replicate across families.** Noah's bar is 3-4; see §5 for the slate.
   Cross-family replication of §2.4 especially.
2. **Second seed.** Every number is seed 1234. Base-specificity must be shown
   not to be a k-means artifact. One reseeded sweep at one rung would do it.
3. **A real baseline comparison.** We have measured-vs-leverage-probe and
   measured-vs-uniform. A Hessian-style proxy (HAWQ-like) would be the
   comparison a reviewer asks for.
4. **Generation evidence.** Every number here is KL/perplexity against a
   teacher. Nothing in this arc has generated a token — see the open gate in
   the GLM ledger. A quality claim without it is incomplete.

Honest read: (1) and (2) are the difference between "interesting internal
result" and "publishable". Neither is expensive.

---

## 4. Scope: is any of this VQ-specific?

Open, and the most consequential question about the work. Three parts, which
have different answers:

**The method is NOT VQ-specific.** Substitute one layer's quantization,
measure, solve a knapsack. Any per-layer quantization choice fits — affine
bit-width per layer is the same problem and is the existing mixed-precision
literature. Architecturally it needs only repeated blocks with independent
quantization choices, which every family we work with has.

**The cheapness is partly VQ-specific, and NOT in our favour.** VQ can only
sweep geometries whose codebooks have already been fitted, so the donor set is
bounded by what we have paid to fit. Affine requantization is nearly free, so
an affine sweep should be CHEAPER than ours, not dearer.

**Whether the FINDINGS are VQ-specific is unknown.** This is the one that
matters. If base-specificity (§2.4) and non-monotonicity (§2.3) hold for
affine too, they are properties of quantized transformers and the claim
touches a far larger literature. If they are VQ-only, it is a narrower result
about codebook error structure. Either answer is publishable; not knowing
which is not.

### 4.1 The affine test (cheap, decisive, not yet run)

GLM's affine ladder (q3 129 GiB / q4 166 / q6 239) is already on disk. Splice
one layer between q3 and q4, measure, then test transfer across bases exactly
as §2.4 did for VQ.

CONFOUND, raised by Noah 2026-08-31 and it is a real one: the affine ladder
steps ~1.0 bpw (q3 3.524 / q4 4.524 / q6 6.524) where our VQ levels step ~0.5
(2.50 / 3.00 / 3.50), and affine's collapse sits ~1 bpw HIGHER than VQ's --
q3 is already collapsed at KL 377 where VQ at 3.58 bpw scores 94.5. So a
q4->q3 demotion pushes that layer across the cliff, which is analogous to our
d8 demotions (~-10 mnats each) rather than to any above-cliff move. A NULL
result on affine could therefore be explained by step size rather than by
anything about affine, leaving the question open rather than answered.
Mitigation if the test is run: use q4->q5 (mlx affine supports intermediate
bit-widths, and group_size gives finer control still) so the step is
comparable, rather than the ladder rungs we happen to have built.

ONE ENGINEERING OBSTACLE, measured 2026-08-31: the affine artifacts share all
2998 keys but NOT their shard assignment — 2654 keys sit in different shards,
because differing bit-widths change tensor sizes and therefore sharding. The
VQ artifacts shared a layout only because they were all built from one struct
base. So the splice tool's `weight_map` equality assertion must be relaxed to
a per-key lookup with an index rebuild. ~20 lines, and the assertion exists
for good reason, so it should be relaxed carefully rather than deleted.

## 5. Replication plan for a paper

Noah's target, 2026-08-31: at least 3-4 families.

| family | status | notes |
|---|---|---|
| GLM-5.3-Flash | done (1 seed) | all findings here come from it |
| Qwen3.8-Flash-Next | ladder exists | VQ rungs published; hybrid attention |
| Qwen3.5-397B | ladder exists | where the published paper's claims live |
| DeepSeek-V4-Flash | not started | never quantized here; needs a readiness pass first |

Plus, orthogonal to family count: a second SEED at one rung (§3.2), and the
affine test above. The seed check and the affine test are each cheaper than
any one family and constrain the claim more.

---

## 6. Relationship to the V2 models

The measured gains are real but modest against rung-to-rung steps: ~+21 mnats
byte-neutral at the 116 rung, ~+55 for +3.4 GiB at the 98.55 rung. The value
for V2 is that byte-neutral gains create headroom that can be spent elsewhere
— e.g. the ~2.1 GiB MTP head, if the parallel MTP work produces a runnable
one. That framing is the reason to develop this, and it is worth stating
that the headroom argument is currently a PLAN, not a measured result.

---

## 7. Two more axes of non-transfer (2026-09-01, overnight run)

The overnight per-rung sweeps produced a WORSE 116 artifact than the earlier
build made with knowingly mistransferred priors. Diagnosed, and it is two
compounding failures — both worth recording because both were mistakes I made
on purpose, for good-sounding reasons.

### 7.1 Sensitivity is TOKEN-SCALE-specific too

Sweeps were run at 512 tokens to save time (ranking "needs less precision
than a ladder score"). The resulting artifact:

| evaluated at | base KL | optimum KL | d_KL |
|---|---|---|---|
| 512 tok (its own scale) | 284.19 | 264.14 | +20.05 |
| 2048 tok (the ladder) | 199.53 | 195.64 | **+3.89** |

Most of the gain does not survive the change of evaluation window. Same
failure mode as §2.4 base-specificity, along the corpus axis: sensitivity
measured on one sample partly fails to transfer to another.

**Rule: sweep at the token count you will be judged at.** The 4x saving is
not worth it — and note the saving was small anyway (~1.5 -> ~1.4 min/sample),
because the splice dominates, not the score.

QUANTIFIED 2026-09-01 by re-running the SAME 42 demotions at both scales,
same base, same corpus — only the evaluation window differs:

| 512 vs 2048 agreement | value |
|---|---|
| R^2 | **0.438** |
| sign flips | 7/42 |
| layers scored as "helps" | 14 @512 vs 11 @2048 |

So a 512-token sweep explains 44% of the 2048-token variance. It also
SYSTEMATICALLY OVERSTATES magnitudes: L6 -22.56 -> -4.96, L40 -16.51 ->
-3.71, L9 -9.79 -> -1.68. Fewer positions means each token disagreement
weighs more, inflating apparent effects. That inflation is what selected L6
and L40 as donor targets for the K16384 fit — and when promoted at 2048 they
measured +3.50 and -2.08, i.e. the second pick actively hurt.

### 7.2 Realisation degrades with DEMOTION fraction

Additive-prediction realisation across every solve so far:

| solve | changes | demotions | realisation |
|---|---|---|---|
| best-8, 98.55 base | 8 | 0 | 0.84 |
| DP optimum, 98.55 base | 19 | 7 | 0.68 |
| r116opt2, 116 base | 19 | 10 | **0.17** |

Promotions compose roughly additively; demotions do not. A demotion-heavy
allocation should be treated as barely predictable, and the solver's objective
badly overstates it. Any future solve should either penalise demotions
explicitly or verify demotion-heavy solutions before trusting them.

### 7.3 The uncomfortable comparison

Best 116-class artifact remains the one built from PRIORS MEASURED AT THE
WRONG BASE (KL 178.33), beating the carefully swept one (195.64). That is not
evidence the careful method is wrong — the careful one was crippled by 7.1 and
7.2 — but it is a caution against assuming rigour in the procedure implies
quality in the artifact. Verify the artifact; the procedure is not the claim.

### 7.4 Splice donors need not share a shard layout

The splice tool originally asserted `weight_map` equality between base and
donor. That held only because the GLM VQ artifacts were all built from ONE
struct base, so every geometry sharded identically. It breaks for:

- a PARTIALLY fitted donor (only some layers at the new geometry — the rest
  keep their old sizes, so shard boundaries move), and
- the affine ladder (§4.1: 2654 of 2998 keys sit in different shards across
  q3/q4/q6, because bit-width changes tensor size).

Fixed 2026-09-01: look each donor tensor up in the DONOR's own weight_map and
load from whatever shard holds it; the OUTPUT keeps the base's layout. The
safety property that actually matters is "every module we intend to splice
exists in the donor", which is now what gets checked. Layout equality was a
proxy for it — a stricter condition than necessary, which is why it passed
everything until it didn't.

This also removes the engineering obstacle §4.1 listed for the affine test.
