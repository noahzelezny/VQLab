# Splice-measured allocation — method note

**Status: developing. One family (GLM-5.3-Flash), one seed. Nothing here is
established until it replicates.** Named 2026-08-31; name not final.

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

1. **Replicate on a second family.** 397B is the natural target — it is where
   the published paper's claims live, and it has the geometry ladder already.
   Cross-family replication of 2.4 especially.
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

## 4. Naming

Working name **splice-measured allocation**, chosen to foreground the swap
trick rather than the knapsack (the knapsack is standard; the cheap exact
measurement is not). Alternatives considered: substitution sweep, direct
sensitivity allocation, codebook reallocation. Not settled — Noah's call.

---

## 5. Relationship to the V2 models

The measured gains are real but modest against rung-to-rung steps: ~+21 mnats
byte-neutral at the 116 rung, ~+55 for +3.4 GiB at the 98.55 rung. The value
for V2 is that byte-neutral gains create headroom that can be spent elsewhere
— e.g. the ~2.1 GiB MTP head, if the parallel MTP work produces a runnable
one. That framing is the reason to develop this, and it is worth stating
that the headroom argument is currently a PLAN, not a measured result.
