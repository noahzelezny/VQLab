# Morning report — 2026-09-09

## Headline: two "open mechanisms" were measurement errors, and the big win is real

The overnight swarms aimed 12 of 27 proposals at two phenomena. Both turned
out not to exist. The arc's largest win was independently verified.

    CB_DEV @ >=16KB   claimed 1.43x   VERIFIED 1.46x   (F32)
    RTILE=64          claimed 1.23x   DOES NOT REPRODUCE, 0.75-0.97x (F25 corrected)
    ragged-NSUB       claimed null    WRONG — it is 1.34x (F31)

**The fleet default is sound.** CB_DEV is what moved ~447 modules to the
device arm yesterday, and it reproduces in a second harness within 2%.

## What dissolved, and why it matters

**There is no RTILE geometry flip.** Matched harness, both artifacts:
35B-3.4 (uniform d4-K2048) 0.75x, Flash-2.1 0.93-0.97x. Flash's milder
number is DILUTION — only 138 of its 272 modules are RTILE-eligible; the
other 128 are d8-K256 on the threadgroup arm and ignore the flag. The
original "flip" compared an exo number against a local one. Also falsified:
prefill chunk size is not the mechanism (2048/4096/8192, sign never moves).
OPEN: exo's 1.23x has not reproduced locally in four runs across both boxes.

**The NSUB=80 "null" is a 1.34x win.** Admitted 834/825 vs refused 615/616
tok/s, refusal instrumented to reject exactly the 552 intended calls. The
original null was measured at 10:29; the artifact's model.py was rewritten at
10:36 — its ADMITTED arm was running a bundle that still refused. Diagnostic
tell: the two harnesses AGREE on refused (601 vs 615) and diverge 40% only on
the arm whose code changed. Flash-2.1 only; the 397B has no ragged modules.

Consequence: five overnight proposals were aimed at explaining a phenomenon
that does not exist, and D8-BENCH's "the null matters more than the win would
have" reasoning is void.

## Proposals verified and killed BEFORE burning GPU time
* **a7** unrunnable — its sweep axis (G) is baked into artifact `scales`
  layout, not an env knob.
* **b4** mis-aimed — its different-kernel mechanism is real but lives in
  `_fused_resolve` (decode); the null it explains was measured in gemmseg
  prefill, which has one source and no shape branching. Surviving new fact:
  on decode, shape B falls off the simd kernel onto the device-cb variant.

## Still genuinely open
1. exo-vs-local RTILE discrepancy (needs an exo-side re-measurement).
2. Decode: 99.83% inside mlx-lm's forward (F23), mechanism unprobed.
3. gemmseg tail-tile waste 8.4% (F24), structural.
4. CB_DEV's occupancy mechanism (F21/F32) — the win is verified, the WHY is not.
5. The 85-minute serving wedge (F27), unreproduced.
6. Publish gate (19 artifacts) — Noah's call.

## Method note that earned its keep
Every correction today came from re-running a claim in ONE controlled
harness. That is now the standing bar before a number enters FINDINGS-LOG.
