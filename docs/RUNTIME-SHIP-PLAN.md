# Runtime ship plan — v2 where weights pay for it, v1.5 everywhere else

**Noah's call, 2026-09-15.** Two runtime configurations ship, and which one a
repo gets depends on whether that repo also gets improved weights.

| name | flags | numerics vs arc6 | speed vs arc6 (35B-3.4, F107) |
|---|---|---|---|
| **arc6** | all off | reference (published 2026-09-09) | reference |
| **v1.5** | OT2, PH2V, D4_WALK, routing memo ON; **both bf16-I/O OFF** | **BIT-EXACT** | +8.1% prefill, +12.2% decode |
| **v2** | all five ON (repo default) | 1-ULP-class; costs up to +0.97% ppl, family-local (F103/F105) | +11.0% prefill, +19.2% decode |

`v1.5` = `VQ_GEMMSEG_BF16IO=0` and `VQ_DECODE_BF16IO=0`; everything else at
its committed default.

## The rule

* **A repo shipping IMPROVED WEIGHTS may ship the full v2 runtime**, provided
  the combined result still beats what that repo serves TODAY on every gate
  corpus. The geometry gain pays for the runtime's accuracy cost, and the
  user gets a model that is both better and faster than the one they have.
  Show the comparison as *new weights on new runtime* vs *old weights on
  arc6* — that is the only pairing a downloader actually experiences.
* **A repo shipping UNCHANGED WEIGHTS ships v1.5.** With no quality gain to
  offset it there is nothing to justify a numerics regression, however small.
  v1.5 is strictly better than what they have: same outputs, meaningfully
  faster.

## Current assignment

| repo(s) | weights | runtime | status |
|---|---|---|---|
| Qwen3.8-Flash-Next-VQ-2.1bpw | **v2 (F100)** | **v2** | gate-complete, UNPUBLISHED |
| the other 19 | unchanged | **v1.5** | 9 of 19 gated (F106); 10 cluster rungs pending |

Flash-2.1 v2 on full v2 vs v1 on arc6 — the downloader's comparison, 12k,
one harness:

| | prose | code | literary |
|---|---|---|---|
| v1 weights / arc6 (today) | 5.8130 | 1.7249 | 7.8201 |
| **v2 weights / v2 runtime** | **5.6710** | **1.7071** | **7.6810** |

Better on all three, plus +11.0% prefill and +19.2% decode. Passes the rule.

## Mechanics

v1.5 is a FLAG DEFAULT, not a separate source tree — the bundled `model.py`
is the same current `vq_switch.py` either way. Shipping v1.5 means flipping
the two bf16-I/O defaults in the bundle. Do NOT ship it as an env var a user
has to set: the artifact must be correct as downloaded.

Per-family flag tuning (turning off only the flag that is numerics-active for
that family, F105) is NOT part of this plan. It buys back a few percent and
costs a per-family bisect before every release; revisit only if the 2.7%/6.2%
ever matters.
