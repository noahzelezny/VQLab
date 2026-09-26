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

### 6.0 STANDING CONVENTION (Noah, 2026-09-01): reserve head room in every rung

Every V2 rung is sized so that PUBLISHED SIZE = TRUNK + MTP HEAD, with the
head's bytes reserved even before a head exists for that family.

Rationale, in his words: a downloader who is excited that a model fits their
machine and then has no room left for a ~50% speedup is a bummer. V1 models
were sized trunk-only, so adding a head later would push them over the
machine class they were chosen for. Reserving up front means the head — and
the VQLab adapter that drives it — can arrive without disqualifying anyone
who already committed to the rung.

Current reservation: 2.9 GiB, from the measured GLM head (7.43B params at
layer 45) quantized at 3-bit + scales = 2.81 GiB. 3-bit is safe because HEAD
PRECISION IS QUALITY-FREE: the trunk verifies every drafted token, so a worse
draft costs a rejection, never a wrong token (VQLab docs/MTP.md §6, measured
-0.2pp acceptance and 0.5% speed at 3-bit vs 6-bit). The head is also
RUNG-INDEPENDENT — grafted from upstream bf16, not derived from the trunk's
quantization — so one head file serves every rung and its cost is paid once.

Caveat to carry: as of this date NO GLM MTP head exists. VQLab has qwen4_exp
implemented and four families identified but unimplemented. So the reservation
is currently a hole for something not yet built. Model cards should either say
so or stay silent about the head until it ships.

### 6.1 Quality headroom

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

---

## 8. The benchmark was wrong (2026-09-01) — supersedes §7.2's prescription

§7.2 concluded that demotion-heavy solves "realise worst" (0.17 vs 0.84) and
prescribed discounting and capping demotions. Phase 6 applied that fix and
produced a NULL result, while the uncapped build it was meant to improve on
produced the night's largest win:

| build | GiB | KL | uniform at that size | advantage |
|---|---|---|---|---|
| 134 base | 134.00 | 94.54 | 94.54 | — |
| r134opt3 (discounted, capped at 6) | 132.52 | 103.27 | 103.31 | **+0.04** |
| r134opt2 (512-tok, uncapped, 17 demotions) | 124.72 | 130.78 | 149.52 | **+18.74** |

**THE ERROR WAS THE BENCHMARK, NOT THE SOLVER.** Realisation-against-additive-
prediction measures whether the MODEL is right. It says nothing about whether
the ARTIFACT is good. The benchmark that matters is uniform scaling at the
size actually achieved — a rung fitted uniformly to that many bytes.

Mechanism: advantage over uniform requires DISPLACEMENT from the base
allocation. r134opt2 moved 9.28 GiB and had room to differ from uniform;
r134opt3 moved 1.48 GiB and so matched uniform almost by construction. The
cap suppressed precisely the thing that generates the gain.

REVISED PRESCRIPTION:
- Objective: maximise predicted gain subject to a SIZE TARGET, not
  byte-neutrality-the-model-believes.
- Do NOT penalise demotions. They are how displacement is bought.
- Expect low realisation on aggressive solves and do not treat that as
  failure — verify the artifact against uniform-at-size instead.
- Keep §7.1 (sweep at the evaluation token count). That one still holds:
  it is about MEASUREMENT validity, not objective design.

SECONDARY FINDING — saturation: at the 134 rung, promotions to K16384 return
4.76 mnats/GiB against a 5.92 uniform rate, i.e. more codebook buys less than
more bits-everywhere. Allocation gains are largest where the rung is far from
saturation (98.55 rung: 1.93x uniform) and vanish where it is close.

---

## 9. Did it actually improve anything? (2026-09-01, honest accounting)

Earlier "vs uniform" figures in this document used LINEAR interpolation
between uniform rungs. The uniform curve is convex, so that overstates
uniform's damage and flatters the method. Against a proper fit of the four
uniform rungs actually built and scored at 2048
(ln KL = 9.5212 - 0.03688*GiB, R^2 0.9972):

| artifact | GiB | KL | uniform @ size | verdict |
|---|---|---|---|---|
| 102 DP-optimized | 101.94 | 291.69 | 318.03 | **+8.3%** |
| r116opt | 116.28 | 178.33 | 187.42 | **+4.8%** |
| r134opt2 | 124.72 | 130.78 | 137.29 | **+4.7%** |
| r116opt2 | 115.85 | 195.64 | 190.41 | -2.7% WORSE |
| r134opt3 | 132.52 | 103.27 | 102.98 | -0.3% null |

THREE REAL GAINS OF 4.7-8.3%, one regression, one null. That is the size of
the effect: worth having at matched bytes, not transformative. Note two of
the three winners came from runs that were methodologically sloppy (wrong-base
priors; 512-token sweep, uncapped) — see §8 on why that is not the paradox it
looks like.

Also note §2.2's headline "1.93x uniform" was computed the same flawed way and
should be read as directional, not exact. The RANKING of methods it reports
(measured >> proxy) is unaffected — that comparison was between two
allocations at IDENTICAL size, where no interpolation is involved.

---

## 10. The 397B v2 arc (2026-09-05) — what transferred, what was new

Full record: `research/qwen397b/V2-SWEEP-PLAN-2.2.md` (+ TSVs). Result:
iso-size v2 at 100.964 GiB, prose +0.0838, code even, shuffled control
loses by 0.077. What this arc TAUGHT, so nobody re-learns it:

**Confirmed a third time (now a law, not a finding):**
- Layer effects are base-specific. The 2.4-base best-6 was the WRONG set
  at the 2.2 base (2.4's #1 pick, L40, was mid-pack). A sweep table
  transfers to NO other rung. Budget the re-sweep; do not argue with it.
- Measured selection >> any proxy, and the shuffled control at identical
  bytes is non-negotiable — ours landed at base+0.007 while the measured
  set took +0.084 from the same bytes.
- Additivity holds at ~76% (composed/sum-of-singles), consistent with
  GLM. Project composed gains at ~0.75-0.8x the single-layer sum.

**New this arc:**
- **Cross-dim splicing works and is now a tool** (v2_sweep --donor
  d4k256). A d8 base takes d4 promoted modules; vq_modules is per-module;
  the runtime dispatches per tensor. Promotion levels can come from any
  SHIPPED rung — zero new fitting for the promotion side, ever again.
- **Demotion-by-measurement works** and is NOT the refuted "downgrade
  the probe's cold layers" move (Flash L-note): candidates were chosen
  by measured promotion-insensitivity, then each demotion was scored
  individually. Two of ten (L24, L31) demoted for FREE — richer codes
  were pure waste there. Free demotions are the iso-size method's fuel.
- **Refit noise is honest cost.** The affine skeleton was deleted, so
  demotion fits re-derive scales from bf16 (demote_fit.py, fit-moe math
  verbatim). Vintage gate: refit-at-shipped-K scored base -0.0037 — in
  family. Whatever ships carries the refit, so no correction is applied.
- **Archive every fit** (HDD vqlab-fits/, [[fit-archive-convention]]).
  The control rebuild hit 6/6 archived fits and took ~25 min instead of
  2.5 h. Candidates are disposable; fits are not.
- **Selection-corpus hygiene**: after two greedy rounds on the same
  8K-token wikitext prefix, held-out scoring (literary corpus, never
  selected against) is required before shipping any further iteration.

**Costs, for planning the next family:** 57-row promotion sweep ~2 h
(65 s/row streaming score at 101 GiB); one demotion fit ~20 min/layer
(d8 K4096, M4); compose chain = fits + seconds of splice. The bottleneck
is k-means, and it is only paid when the archive misses.

---

## 11. The v3 exploration (2026-09-06) — two closed doors and a noise floor

Data: `research/qwen397b/overnight-v3.tsv`. Nothing here changed what
ships; all three results are things not to spend money on again.

### 11.1 Deep donors lose. Promote SHALLOW and WIDE.

Off the 2.2 base (d8/K16384, 1.75 bpw), promoting one layer deeper costs
+0.3743 GiB at d4/K512 and +0.75 at d4/K2048, against +0.1868 at d4/K256:

| layer | donor | gain | cost GiB | gain/GiB |
|---|---|---|---|---|
| L43 | K256 | +0.0342 | 0.187 | **0.183** |
| L43 | K512 | +0.0402 | 0.374 | 0.107 |
| L43 | K2048 | +0.0601 | 0.750 | 0.080 |
| L45 | K256 | +0.0219 | 0.187 | **0.117** |
| L45 | K512 | +0.0349 | 0.374 | 0.093 |

Doubling a hot layer's codebook bought 18% more gain for 100% more
bytes. Returns to depth are steeply sublinear; returns to BREADTH (more
layers at the cheapest step) are near-linear until the layer pool runs
out of hot candidates. The iso-size recipe's shape — six shallow
promotions — is therefore the right shape, not merely a convenient one.

### 11.2 Single-layer effects have a ~0.004 (1σ) FIT-NOISE floor.

K8192 is strictly richer than K4096, so its damage must be <= K4096's on
every layer. Measured on seven layers, **three violate that by up to
+0.0088**, and mean damage came out HIGHER at K8192 (+0.0055) than at
K4096 (+0.0041) while saving half the bytes. Since the scorer is
deterministic, the variance is in the FIT — k-means path-dependence
(++ seeding on a 200k subsample, 20 Lloyd steps, a different local
optimum per run) — not in the measurement. Implied per-row sigma ~0.004.

Consequences, all binding:

- **"Free demotions" is RETRACTED.** L24 and L31 read -0.0002/-0.0001 at
  K4096 and were reported as costing nothing. Re-fitting them at K8192
  put them at +0.0080/+0.0087 — the worst two of the batch. They were the
  lucky tail of a noisy draw, not insensitive layers.
- **Do not rank individual rows whose effects are under ~0.01.** That
  covers most of a demotion sweep and the tail of a promotion sweep.
  Sets of top rows remain meaningful; their internal order does not.
- **Expect winner's curse.** Choosing the best 6 of 57 by one noisy fit
  selects partly for luck, so a composed candidate lands below the sum of
  its singles. The 397B v2 composed at 76% of that sum — previously
  charged entirely to sub-additivity; some of it is this.
- **K4096 is the currency of choice.** It saves 2x the bytes of K8192 for
  damage that is statistically indistinguishable at this precision.
- To actually rank near-tied candidates: fit N seeds per layer and keep
  the best, or replicate and average. Both cost N x 20 min/layer.

### 11.3 relerr does NOT predict ppl damage. (Third falsified proxy.)

Pearson(mean fit relerr, measured ppl damage) = **+0.08** (n=10, K4096)
and **+0.23** (n=7, K8192). The two K4096 fits with the HIGHEST
reconstruction error were the two that scored as free. Reconstruction
error is a per-tensor objective; end-to-end damage is not a function of
it. Joins `vqlab layer-leverage` (§2.2) and the leverage probe's cold end
on the list of cheap proxies that do not survive contact with
measurement. There is still no substitute for scoring the assembled
model.

### 11.4 What this does NOT touch

The shipped candidate stands. `397b-v2-iso100` was measured directly, not
inferred: prose +0.0838 with a byte-identical shuffled control at +0.0067
(a gap of ~18 sigma against the floor above), and independently confirmed
on a held-out literary corpus never used for selection (+0.0403 vs the
control's +0.0052). Composed artifacts are measured; only the per-layer
attributions inside them are noisy.

---

## 12. Per-PROJECTION allocation (2026-09-06) — a third of every promotion was wasted

Until now every promotion moved a layer's three expert projections
together, because that is what the tooling did (`PROJECTIONS` was a
hardcoded tuple). `vq_modules` is keyed per MODULE and a module IS a
projection, so this was never a runtime constraint — nobody had written
a config that split them. All three are exactly 2.147B params on the
397B (gate/up map 4096->1024, down maps 1024->4096), so a projection is
an exact third of a layer: **0.0623 GiB instead of 0.1868.**

L43, the hottest layer in the 2.2 sweep, promoted d8/K16384 -> d4/K256
one projection at a time (each 0.0623 GiB, vs +0.0342 for all three):

| promoted | Δ prose | prose/GiB | Δ code | Δ literary |
|---|---|---|---|---|
| gate_proj | **−0.0006** | **−0.010** | +0.0005 | +0.0001 |
| up_proj | +0.0094 | 0.151 | +0.0012 | +0.0023 |
| **down_proj** | **+0.0227** | **0.364** | +0.0015 | +0.0060 |
| whole layer | +0.0342 | 0.183 | — | — |

- **gate_proj buys NOTHING** — it is inside the noise floor and signed
  negative. Every promotion this project has ever made spent a third of
  its bytes on it.
- **down_proj carries 66% of the layer's gain for 33% of its bytes**, at
  **2.0x the byte-efficiency** of promoting the whole layer. It leads on
  all three corpora, and it carries the literary gain almost entirely.
- Decomposition is **92% additive** (singles sum +0.0315 vs +0.0342
  measured together), so composing by projection should behave.

### GENERALITY TESTED (same day): L43 was an OUTLIER. Lever mostly CLOSED.

Three more hot layers decomposed the same way. Every column is a
promotion, so these carry NO fit noise (a splice of shipped 2.4 donor
tensors is deterministic); the only error term is corpus window.

| layer | gate | up | down | sum | whole | additivity | winner |
|---|---|---|---|---|---|---|---|
| L43 | -0.0006 | +0.0094 | **+0.0227** | +0.0315 | +0.0342 | 92% | down |
| L45 | **+0.0081** | +0.0032 | -0.0038 | +0.0075 | +0.0219 | **34%** | gate |
| L29 | -0.0008 | +0.0024 | **+0.0088** | +0.0104 | +0.0175 | 59% | down |
| L47 | +0.0043 | **+0.0085** | +0.0058 | +0.0186 | +0.0183 | 102% | up |

- **No projection is universally dead or universally best.** gate_proj
  looked worthless on L43 and is the WINNER on L45. down_proj is the star
  on L43 (+0.0227) and NEGATIVE on L45 (-0.0038). All three win somewhere.
- **The 2x byte-efficiency does NOT survive.** Averaged over four layers,
  down_proj buys 0.134 prose/GiB against 0.123 for promoting the whole
  layer -- **+10%, not +100%.** The L43 result was one layer.
- **Additivity is unreliable: 34% to 102%.** On L45 the three projections
  sum to a third of what they deliver together, i.e. most of that layer's
  gain is EMERGENT from promoting them jointly and splitting destroys it.
  This is the real blocker: per-module tables cannot be composed by
  addition the way per-layer tables can.
- Best-per-layer picking would give 0.193 prose/GiB (+57% over whole
  layer), but that is the max of three draws per layer and therefore
  winner's-curse inflated, and it needs the full 171-module sweep (~8.5 h,
  no fitting) to know the winners at all.

**Verdict: do not adopt a per-projection rule.** "Promote down_proj only"
is refuted. A measured per-layer-per-projection sweep might still buy
something real, but it triples the selection surface against effects a
third the size, on a family where composition is already sub-additive.
Whole-layer promotion stays the default. Ordering down > up > gate holds
only ON AVERAGE (0.134 / 0.094 / 0.044 prose/GiB) and not per layer.

The mechanism story -- down_proj writes into the residual stream while
gate_proj feeds a SiLU whose saturation absorbs error -- is SPECULATION
and is not supported per-layer by the table above.

---

## 13. Corpus-window variance (2026-09-06) — the last error term, and it is big

Every number in the 397B v2/v3 arc came from ONE 8192-token prefix. Fit
noise is characterised (§11.2, ~0.004 sigma) but applies only to fits;
**promotions are deterministic splices of shipped donor tensors and carry
no fit noise at all**, so window variance is the ONLY error term on them
and it had never been measured.

Six DISJOINT windows of the literary corpus (60 KB apart, ~33 KB
consumed each), three artifacts scored on each:

| window | base 2.2 | iso100 (v2) | v3 | v3−base | v3−iso100 |
|---|---|---|---|---|---|
| w0 (the window used all day) | 1.2820 | 1.2417 | 1.2422 | +0.0398 | −0.0005 |
| w1 | 1.3059 | 1.2559 | 1.2541 | +0.0518 | +0.0018 |
| w2 | 1.9402 | 1.8471 | 1.8356 | +0.1046 | +0.0115 |
| w3 | 1.8557 | 1.7653 | 1.7537 | +0.1020 | +0.0116 |
| w4 | 1.4751 | 1.4095 | 1.4029 | +0.0722 | +0.0066 |
| w5 | 1.5249 | 1.4432 | 1.4337 | +0.0912 | +0.0095 |
| **mean / sd** | | | | **+0.0769 / 0.0269** | **+0.0068 / 0.0051** |

1. **Direction is robust; magnitude is not.** v3 beats base on **6/6**
   windows, but the delta ranges +0.0398 to +0.1046 — a **35% relative
   standard deviation**. Absolute ppl deltas scale with how hard the
   window is (easy w0 at 1.28 gives +0.040; hard w2 at 1.94 gives +0.105),
   so a single-window delta is a point on a wide distribution, not a
   measurement to three decimals. **Quote ranked direction, not
   precision, from a single window.**
2. **Close comparisons do NOT survive.** v3 over iso100 is +0.0068 mean
   with sd 0.0051 — about 1.3 sigma, winning 5/6 windows and LOSING on
   w0, which happens to be the window the whole arc used. The "v3 beats
   v2 by ~2 sigma" claim made from w0's prose row is not supported;
   the honest statement is v3 >= iso100, probably slightly better, and
   its real advantage is structural (no demoted layers, less selection
   surface) rather than numerical.
3. **The window we happened to pick was the least favourable to v3** of
   the six. Not cherry-picked — it was simply the corpus prefix — but it
   is a reminder that a single window can understate as easily as
   overstate.

**Standing rule from here:** any margin under ~0.02 measured on one
window is not a result. Either score several windows, or report it as a
tie. Cost is trivial — six windows is ~6 minutes per artifact — and this
study cost less than one demotion fit.

---

## 14. Affine-side probe (2026-09-07) — embeddings were over-provisioned

First measurement ever taken on the 397B's affine allocation. Null test
first: re-quantizing `lm_head` at its CURRENT 6 bits reproduced the base
to +0.0003 prose / +0.0005 literary, so the tool
(`research/qwen397b/affine_requant.py`) is sound.

| change | Δ size | Δ prose | Δ lit_w2 |
|---|---|---|---|
| lm_head 6 -> 4 | **−0.177 GiB** | **−0.0161** | −0.0093 |
| lm_head 6 -> 8 | +0.237 GiB | +0.0004 | +0.0010 |
| **embed_tokens 6 -> 4** | **−0.177 GiB** | **−0.0003** | **+0.0009** |
| embed_tokens 6 -> 8 | +0.237 GiB | −0.0012 | −0.0010 |

- **`embed_tokens` at 6 bits is WASTE.** Dropping it to 4 costs nothing
  measurable on either corpus (both inside the noise floor, literary
  even signs positive) and refunds **0.177 GiB** — ~2.8 projection-units
  of promotion budget, free.
- **`lm_head` at 6 bits is CORRECT.** Going down to 4 costs a real
  −0.0161 prose (4x the noise floor); going up to 8 buys nothing. The
  original choice was well made, and the two tensors are NOT
  interchangeable despite being the same size and both 6-bit.
- Asymmetry is the finding: identical-looking tensors on the two ends of
  the model have opposite sensitivity. Input embeddings tolerate coarse
  quantization; the output head does not.

Scoped but NOT run: `shared_expert` (180 modules, 6-bit, ~0.57 GiB) and
the linear-attention input projections (90 modules at 4-bit, ~1.98 GiB).
Both span 27 of 28 shards, so each candidate is a ~100 GiB rewrite --
affordable only with substantial free space, and never with another
sweep running. The shared expert is the interesting one: every token
passes through it, unlike a routed expert seen by ~10 tokens in 512.
