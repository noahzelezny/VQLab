# Crush run — Qwen3.8-27B + gemma-4-26b-a4b (2026-08-17 evening)

Affine ladder built on the M3 Ultra 96GB; the VQ fit ran on the M4 Max
128GB (reachable via nozzlebook-pro.local / tailscale — an earlier line here
wrongly said there was no ssh path). mlx_lm 0.31.3, mlx 0.32.0 on both.
Sections 1-3 are the AFFINE baseline; the VQ run that beats it is at the
bottom.

## Qwen3.8-27B (dense, 27.78B, bf16 55.6G)

Instruments: referee wikitext ppl (valid on Qwen) AND KL-to-bf16, together.

| rung | size | ppl | vs bf16 | KL (mnats) | top-1 agree |
|---|---|---|---|---|---|
| bf16 | 55.6G | 5.2249 | 1.000x | 0 | 100.00% |
| **q4** | **14G** | 5.2055 | **0.996x** | 45.8 | 89.82% |
| m3-a4 | 13G | 5.4377 | 1.041x | 106.4 | 85.36% |
| m3-a3 | 12G | 5.6767 | 1.086x | 140.1 | 83.46% |
| **q3** | **11G** | 5.8323 | **1.116x** | 187.8 | 79.48% |
| m2-a4-gs32 | 12G | 6.5318 | 1.250x | 394.5 | 72.44% |
| m2-a6 | 12G | 7.0913 | 1.357x | 495.3 | 69.90% |
| m2-a4 | 11G | 7.0976 | 1.358x | 504.3 | 69.67% |
| q2 | 5.7G | 16.4349 | 3.146x | 1426.9 | 46.07% |

**q4 at 14G is free** (0.996x — 4x compression, no measurable ppl cost).

**Hand-designed mixed allocation LOSES to uniform on this model.** At a
matched 11G, uniform q3 (1.116x) beats mixed m2-a4 (1.358x); m2-a6 at 12G
is still worse than q3 at 11G. Cause is structural: mlp is 61.6% of the
model, so 2-bit mlp IS the dense cliff (EXPERIMENTS.md headline 4) and no
attention protection buys it back. Mirror image of the MoE case, where
experts are ~90% and tolerate 2-bit.

**This does NOT falsify E4.** These are hand-designed STATIC allocations,
not optiq's KL-calibrated ones. A real OptiQ sweep is still untested and
now has a concrete bar: **beat 1.116x at 11G**. Note the bf16-reference
bug that voided the 397B sweeps does NOT fire here — everything is
`nn.Linear` ndim==2 except conv1d, which is 2.0M params (0.01%).

## KL is validated as a ppl proxy

Across all 9 Qwen rungs, KL and ppl rank IDENTICALLY and monotonically
(1.00x -> 3.15x). That is the licence to trust KL on gemma, where ppl does
not exist. Rough dense conversion:

    KL <  50 mnats -> free            (q4: 45.8, ppl 1.000x)
    KL ~ 200 mnats -> ~ +10% ppl      (q3: 187.8, ppl 1.116x)
    KL > 1000 mnats -> broken         (q2: 1426.9, ppl 3.146x)

**These thresholds are DENSE-derived and do NOT transfer to MoE — see below.**

## gemma-4-26b-a4b (hybrid MoE, 25.23B LM, bf16 48G)

Architecture correction found during the run: **every layer has a dense
`mlp.{gate,up,down}_proj` ALONGSIDE `experts.switch_glu.*`.** It is a
hybrid, not a pure MoE. Also `v_proj` exists in only 25 of 30 layers
(`attention_k_eq_v`). Corrected split:

    routed experts  22.838B  90.5%
    attention        1.110B   4.4%
    embed/lm_head    0.738B   2.9%
    dense mlp        0.535B   2.1%   <- runs on EVERY token
    router           0.011B   0.0%

So non-expert at full 8-bit costs only 2.54G, leaving 2.05 bpw for experts
at an 8.4G target. Structure is cheap; all pressure is on experts.

| rung | size | KL (mnats) | top-1 agree |
|---|---|---|---|
| struct8-e8 | 25G | 441 | **79.95%** <- practical ceiling |
| uniform-q8 | 25G | 472 | 79.33% |
| struct6-e8 | 24G | 2959 | 45.20% |
| struct6-e6 | 19G | 2994 | 44.18% |
| struct6-e4 | 14G | 3180 | 42.73% |
| uniform-q4 | 13G | 4701 | 30.32% |
| struct6-e3 | 11G | 3734 | 38.01% |
| uniform-q3 | 10G | 15603 | 4.46% |
| **struct8-e2** | **9.1G** | **4648** | **34.90%** <- best at target size |
| struct6-e2-qkv8 | 8.8G | 4823 | 33.68% |
| struct6-e2-qkv6 | 8.5G | 4939 | 33.28% |
| struct6-e2 | 8.3G | 5810 | 27.29% |

### What this shows

1. **The predicate is correct.** struct8-e8 (441/79.95%) reproduces
   uniform-q8 (472/79.33%) and is marginally better — the bf16 router.
   Module re-targeting (`switch_glu`, `router.proj`) is verified against
   real tensors.
2. **The structured recipe beats uniform, decisively, at every small size.**
   struct6-e3 (11G, 38.0%) beats uniform-q4 (13G, 30.3%). struct8-e2
   (9.1G, 34.9%) beats uniform-q3 (10G, 4.5%) by a mile.
3. **Attention is the cliff, again.** struct6-e8 collapses to 45.2% purely
   because qkv sat at 4-bit — worse than uniform-q8's 79.3% despite a bf16
   router. Raising qkv/structure to 8-bit recovers it. E8 reproduced on a
   new family.
4. **Expert bits barely matter above the attention floor**: e8 45.2 -> e6
   44.2 -> e4 42.7 -> e3 38.0 -> e2 27.3. The floor is set elsewhere.
5. **8-bit is gemma's practical ceiling at only ~80% agreement**, where
   Qwen reached 89.8% at 4-bit. Near-lossless quantization still reads as
   large KL here. Likely inherent to MoE: routing is DISCRETE, so any
   upstream perturbation flips which 8-of-128 experts fire and the output
   changes categorically. **Therefore the dense KL thresholds above must
   NOT be applied to gemma.** Measure gemma damage RELATIVE to the ~441
   mnats / 80% 8-bit reference, not against zero.
6. **35% agreement is not a broken model.** struct8-e2 (9.1G) still
   generates coherent, on-task chat output; it is visibly thinner than
   struct8-e8, not incoherent.

### The bottom line for the sidecar goal

Target was 26b at ~8.4G (e4b-8bit's size). **Affine gets to 9.1G at 34.9%
agreement against an 80% ceiling.** That gap — 80% down to 35% — is exactly
what VQ exists to close, and is the same shape as the 397B result where
affine 2-bit experts were poor and VQ at matched size was not.

**Next: VQ the experts.** Everything non-expert stays at 8-bit (only
2.54G), experts go to VQ at ~2.0 bpw. Preflight blocker still open:
`down_proj` cannot sub-byte pack at d=4 (moe_intermediate 704 -> NSUB 176,
176 % 32 != 0). See LADDER_GEMMA.md for the three options.

---

## VQ RUN (later the same evening) — the gap closes

`vq_397b_codes.py --family gemma4 --k 256 --dim 4`, fitted on the M4 Max in
315s over the struct8-e2 base (experts 2-bit affine) with the bf16 as value
source. 90 tensors, **mean relerr 0.3136** — statistically the same as the
397B fit's 0.3156 (EXPERIMENTS.md:1519), so the fit is healthy on a new
family rather than struggling.

| rung | size | KL (mnats) | top-1 agree |
|---|---|---|---|
| struct8-e8 (affine) | 25G | 441 | 79.95% (ceiling) |
| struct6-e4 (affine) | 14G | 3180 | 42.73% |
| **VQ K256 d4** | **8.4G** | **3363** | **42.65%** |
| struct6-e3 (affine) | 11G | 3734 | 38.01% |
| struct8-e2 (affine) | 9.1G | 4648 | 34.90% |

**VQ at 8.4G equals affine at 14G** (42.65% vs 42.73%) — same quality for
40% of the bytes — and beats the affine rung at its own size by +7.8 points
of agreement while being 0.7G SMALLER. This is the 397B result reproducing:
affine 2-bit experts are poor, VQ at matched size is not.

It also hits the sidecar target exactly: **8.4 GiB = gemma-4-e4b-it-8bit's
8.4G**, with audio still graftable.

Still open: 42.65% sits well under the 79.95% 8-bit ceiling. Whether that
ceiling is reachable at ~8G, or is a floor imposed by discrete MoE routing,
is the next question — K/d geometry (E36: down_proj prefers larger d) and a
tail schedule are the untried levers. Packing is a separate, final, safe
pass (see LADDER_GEMMA.md; per-module skip is the clean option).

## FLOOR TEST — what "79.95%" actually means

Prompted by the 397B session, who showed on their lineup that per-item
disagreement is SYMMETRIC (on disagreeing items neither model matches gold
more often — 29v31, 16v18, coin flips) and concentrated on NEAR-TIES (top-2
margin 1.2-2.4 nats on disagreements vs 16-18 nats on agreements, up to
14.7x). I.e. quants reshuffle the near-ties and leave the confident mass
alone, so agreement overstates quality difference.

So: measure two models we have INDEPENDENTLY established are equivalent
against each other, and whatever that reads is the metric's floor.

| comparison | KL (mnats) | top-1 agree |
|---|---|---|
| struct8-e8 vs ITSELF (control) | 0.000 | 100.00% |
| **struct8-e8 vs uniform-q8** | **397** | **82.32%** |
| struct8-e8 vs bf16 | 441 | 79.95% |
| VQ K256 d4 vs bf16 | 3363 | 42.65% |

**The floor is ~82%.** Two near-lossless 25G artifacts, equivalent by every
other measure, disagree on 17.7% of positions. The self-comparison control
returns exactly 100.00% / 0.000, so this is not instrument error — it is
genuine near-tie reshuffling, on a model whose distribution is collapsed
(GEMMA4_PPL_ANOMALY.md) and whose routing is discrete.

**CONSEQUENCE 1 — restate the ceiling.** "79.95% vs bf16" was never a
quality ceiling; it is 8-bit sitting essentially AT the metric's floor,
which is the correct reading for near-lossless quantization. The metric
cannot distinguish "identical quality" from "8-bit" and should not be asked
to.

**CONSEQUENCE 2 — VQ's gap is REAL, but NOT for the reason first written
here.** The original argument (and the peer's own decision rule, which they
retracted) was "a low floor means real damage". That is wrong: the floor
tells you only where the TOP of the scale sits — that near-lossless reads as
82%, not 100%. It says nothing about whether the region 40 points below has
resolution.

What actually licenses the conclusion is the AFFINE LADDER already measured
above: struct6-e2 27.29 -> struct8-e2 34.90 -> struct6-e3 38.01 -> struct6-e4
42.73. The metric cleanly separates four rungs across the 27-45 band, which
is exactly where VQ sits at 42.65. **A saturated instrument cannot do that.**
So the metric demonstrably has resolution in the working region, and
improvements there will be visible. That is why K/d geometry and a tail
schedule are worth the compute.

**QUOTE KL, NOT AGREEMENT, FOR THIS CLAIM.** The floor shows up more cleanly
in KL: 397 (two equivalent artifacts) vs 441 (near-lossless vs source) are
essentially the same number, so KL's noise floor is ~400 mnats and VQ at
3363 sits at **~8.5x the instrument's noise floor**. That statement does not
require the reader to know the ceiling is 82%.

**METHODOLOGY FACT — the two metrics are complementary and neither is
trustworthy alone.** Agreement SATURATES near the top (it cannot distinguish
"equivalent" from "8-bit": 82.32 vs 79.95) but DISCRIMINATES lower down
(27->43 cleanly). KL behaves the opposite way: it still separates 397 from
441 where agreement has given up, and it compresses differences lower down
into large hard-to-read numbers. Report both; trust agreement in the damaged
region and KL near lossless.

Caveat kept from the peer: their evidence is 4-choice argmax on tasks, ours
is top-1 over a ~250k vocab on free text. Comparable PHENOMENA, not
comparable numbers. What transfers is the mechanism (near-tie reshuffling),
not the percentages.

## K/d GEOMETRY — E36 does NOT transfer to gemma

E36 (397B) found `down_proj` prefers larger d — **but E37 ALREADY FALSIFIED
THAT** as a layer-0 probing artifact (d8 relerr climbs 0.1793 at L0 to 0.4148
at L56; the mixed artifact lost on both corpora, relerr 0.3474 vs 0.3156).
This run cited E36 without reading E37, so it re-derived a known result. It
stands as independent SECOND-FAMILY corroboration of E37 — same direction,
same shape — but that is luck, not method. Gate/up held at d4k256,
`down_proj` moved to d8k256:

| rung | size | KL (mnats) | top-1 agree | mean relerr |
|---|---|---|---|---|
| **VQ K256 d4 (all)** | 8.4G | **3363** | **42.65%** | 0.3136 |
| VQ K256 d4/d8 | 7.5G | 5689 | 29.88% | 0.4010 |

**Worse on every axis except size**, and the fit itself says so before any
scoring: relerr 0.3136 -> 0.4010.

IMPORTANT — this is NOT a matched-size comparison, and the reason matters.
At fixed K, d8 stores 8 bits per EIGHT weights where d4 stores 8 bits per
FOUR: 1.25 bpw vs 2.25 bpw (the fitter prints this). So d8 halves the bit
budget for that projection. Matching bpw at d=8 would need K=65536, which is
absurd — d8 is inherently a bits-for-subvector-size trade, and on gemma the
trade is bad. Read the result as "d8k256 down_proj is a bad operating point:
it saves 0.9G and costs 12.8 points of agreement", not as "d8 is refuted at
matched budget".

Practical consequence: **the all-d4 K256 artifact at 8.4G remains the best
gemma rung**, and the packing skip for `down_proj` stays motivated purely by
`176 % 32 != 0` rather than by geometry (see LADDER_GEMMA.md — the two were
consistent, but only one is now load-bearing).

Untried lever remaining (and the one E37 also points to): a TAIL SCHEDULE (more expert bits in some layers,
397B's struct6-tailN shape), which spends bytes where they help instead of
uniformly. That is the next thing to run, not more geometry.

## CHAT-NATIVE COMPARISON — first attempt INVALID, do not cite it

litbench scored through each model's own chat template with lettered options,
reading the answer-letter logprob at the first generated position. Position
bias controlled with --cyclic. Results:

    e4b bf16      78.85%        vq-tail10   49.04%
    struct8-e8    44.23%        vq-K256-d4  38.46%
    struct8-e2    32.69%        26b bf16    37.50%

**THESE NUMBERS ARE AN ARTEFACT AND MUST NOT BE QUOTED.** Two tells: 26b
bf16 (37.5%) scored BELOW its own 8-bit quant (44.23%), which is impossible
if the metric measures quality; and the 25G near-lossless rung landed at 44%
while a 19G e4b hit 79%.

ROOT CAUSE — single-token scoring measures WILLINGNESS TO ANSWER
IMMEDIATELY, not comprehension. Top-5 next tokens after the generation
prompt:

    e4b bf16 : '<|channel>', 'D', 'A', '$', 'C'          <- letters present
    26b bf16 : '<|channel>', '<', '---', ' <', ' inner'  <- NO letter at all

Both open a thinking channel; the 26b is more committed to it and writes a
genuine analysis of every option before answering. Reading letter logprobs
at that position penalises the better reasoner. Confirmed by generation: the
26b spends 200+ tokens correctly reasoning about the coral-reef metaphor and
had not reached its answer when the budget ran out.

FIX — `litbench_chat.py --generative`: let the model think, parse the answer
letter after the `<channel|>` close. Same model, same items, 6-item smoke:

    26b bf16 single-token  37.5%  ->  generative  100%

**Single-token mode stays valid for comparing QUANTS OF ONE MODEL** (both
sides share an answering style, so the bias cancels) and is invalid across
models. Generative sweep running; those are the numbers to use.

LESSON, and it is the same one twice tonight: an instrument that is
in-distribution for one model can be out-of-distribution for another. Raw
continuation broke gemma-vs-Qwen (E39); single-token chat broke
e4b-vs-26b. Both times the artefact looked like a decisive capability gap.

## OptiQ ON QWEN3.8-27B — CALIBRATION LOSES TO UNIFORM, AND THE FLOOR LOSES HARDER

The E4 hypothesis ("calibrated allocation beats uniform on DENSE at matched
budget in the steep zone") tested properly this time: real optiq
`--method optiq --reference bf16`, not a hand-designed static allocation.
497 components profiled, ~8h on the M3 Ultra.

| rung | size | ppl | vs bf16 | KL (mnats) | top-1 agree |
|---|---|---|---|---|---|
| uniform q3 (baseline to beat) | **11G** | 5.8323 | **1.116x** | 187.8 | 79.48% |
| optiq-b30 (unfloored) | 13G | 6.1596 | 1.179x | 302.1 | 78.57% |
| optiq-b30-af6 (attn floor 6) | 13G | 8.4690 | 1.621x | 681.4 | 61.04% |

**BOTH LOSE TO PLAIN UNIFORM q3 — while being 2G LARGER.** Calibrated
allocation is not merely failing to win here; it is paying more bytes for
worse quality. E4 does NOT replicate on Qwen3.8-27B.

**The attention floor makes it much worse, not better.** Predicted before
scoring, and the reason is already in this file: the floored allocation is
`full_attn` all-6, `linear_attn` all-6, `mlp` all-2 — which is essentially
the hand-designed `m2-a6` rung that scored 1.357x earlier tonight. Forcing
6-bit attention leaves the allocator no budget anywhere else, so it crushes
100% of the mlp to 2 bits, and mlp is 61.6% of a dense model. Same cliff,
reached by a different road.

What unfloored calibration actually chose (and it is NOT the proven recipe):

    full_attn    2:1  3:17  4:25  6:21   <- genuinely assigns 2- and 3-bit attention
    linear_attn  2:10 3:66  4:64  6:4
    mlp          2:78 3:104 4:6   6:4

So the flat-per-layer-KL problem that motivated `OPTIQ_ATTN_FLOOR_BITS` on
MoE is NOT MoE-specific: on a dense model too, isolation-KL rates attention
insensitive enough to hand some layers 2 bits. But flooring it is not the
cure either, because on a dense model the bytes have to come out of the mlp
bulk, which cannot afford them.

**CONCLUSION FOR THE LADDER: use uniform for Qwen3.8-27B.** q4 at 14G is
free (0.996x); q3 at 11G costs 11.6%. Nothing tried tonight — hand-designed
static mixed, optiq calibrated, optiq calibrated+floored — beat plain
uniform at matched or smaller size. Three independent attempts at
mixed-precision on this model, all losing, is a strong result rather than a
null one.

Cost note: the af6 variant took **3 minutes**, not the ~10h a naive re-run
would have cost, because optiq resumes from `sensitivity_checkpoint.json`
(core/sensitivity.py:870) when candidate_bits match. Copy the checkpoint into
the new output dir first. Sensitivity is a property of model+calibration,
not of the bit budget.
