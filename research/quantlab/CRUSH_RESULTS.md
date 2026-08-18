# Crush run — Qwen3.8-27B + gemma-4-26b-a4b (2026-08-17 evening)

Single box: M3 Ultra 96GB (no ssh path to the M4). All rungs built with
mlx_lm 0.31.3 affine quantization. **No VQ yet** — this is the affine
baseline that VQ has to beat.

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
