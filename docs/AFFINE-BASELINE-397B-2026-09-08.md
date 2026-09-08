# The VQ prefill tax at 397B scale — measured at last (2026-09-08)

`MTP.md` line 180 has said "The VQ prefill tax remains unquantified" since it
was written, and named exactly this comparison as the way to settle it. The
only affine reference numbers in the repo were for the **35B** (an affine-8bit
floor of 2246 tok/s), which is a different model class and cannot answer what
VQ costs on the flagship.

Both sides now measured on the same M3+M4 ring, same prompts, same harness.

## Numbers

| | affine 2.6bit | VQ 2.2bpw | VQ / affine |
|---|---|---|---|
| prefill 1543 tok | 250 tok/s (6.171 s) | 228 tok/s (6.761 s) | **91%** |
| prefill 6919 tok | **411 tok/s** (16.816 s) | 338 tok/s (20.452 s) | **82%** |
| decode 200 tok | 29.8 tok/s | 27.9 tok/s | 94% |
| size on disk | 121 GB | 112 GB | 0.93x |

4 reps plus an untimed warm pass per length; min reported, spreads 3.9-5.5%.
Artifacts: `spicyneuron/Qwen3.5-397B-A17B-MLX-2.6bit` (`mode: affine`,
bits 2, group_size 64) vs `TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw`
(fused d8+d4). Same architecture both sides: head_dim 256, 60 layers,
512 experts.

**VQ prefill is at 82% of affine at 9k context.** The remaining tax is
~1.22x, not the ~1.5x the 35B numbers had us assuming.

## Caveats — read before quoting this

1. **Not bit-matched.** 2.6bit affine vs 2.2bpw VQ. VQ moves ~8% fewer bytes,
   which flatters it on memory traffic and penalises it on decode work. A
   same-bpw pair would be cleaner; this is the affine artifact that exists.
2. **MTP asymmetry.** The affine run had MTP OFF (no sidecar exists for
   spicyneuron); the VQ run had it ON. That inflates VQ's DECODE number and
   does not touch prefill — so the 82% prefill figure is the solid one and
   the 94% decode figure is soft. Do not quote the decode ratio without this.
3. **Separate loads, not interleaved** — flipping between artifacts needs a
   placement change. Same methodological limit as the d8 bench.
4. The tax narrows at short context (91% at 1.5k vs 82% at 6.9k), i.e. it
   scales with prompt length, consistent with a per-token decode cost rather
   than a fixed overhead.

## Why this matters for the remaining work

The three refuted swarm leads predict, at their mid-ranges:

- RTILE 32->64 (paired token tiles): +5-8%
- NSUB=80 ragged tail: +4-10% (Flash-2.1 specifically)
- OTILE 32->64: +5-10%, but its own refuter demoted it to second-order

A mid-range RTILE alone moves VQ prefill from 82% to roughly 88% of affine.
The gap is ~22% with three credible attacks on it — it is not the 1.5x
mountain the 35B framing implied.

## Method note on the probe

Measured AFTER removing the dtype probe (exo 15170e22). Worth recording that
the probe's cost was NOT uniform and an earlier claim that it understated
everything was wrong: it fired per pipeline-layer per TOKEN, so decode paid
60 prints/token (GLM 8 -> 17.5 tok/s) while prefill amortised them across a
whole chunk. Re-measured clean, the 397B's numbers moved within noise
(347 -> 338 tok/s prefill, 29.8 -> 27.9 decode) — i.e. slightly DOWN, so the
probe never inflated the ratios and the d8/MTP conclusions stand unchanged.
