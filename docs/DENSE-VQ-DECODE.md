# The dense 27B VQ artifact decodes ~39x slower than stock

**Status: measured and reproduced, cause not yet isolated. Found 2026-09-01
while benchmarking MTP; it has nothing to do with MTP.**

## The measurement

One machine (M4 Max, 128 GB), one harness (`mlx_lm generate`), one prompt, 128
tokens, greedy. No drafting head involved in either run.

| model | prompt tok/s | **generation tok/s** | peak |
|---|---|---|---|
| `Qwen--Qwen3.8-27B-8bit` (stock) | 123.26 | **16.687** | 28.9 GB |
| `TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw` (ours) | 9.31 | **0.426** | 21.5 GB |

**39x slower generation, 13x slower prefill --- while using 7 GB LESS memory.**
Fewer bytes of weight and far less throughput is not a bandwidth story; the
work is being done somewhere other than the matmuls.

## It reproduces, and it is not the storage link

The first sighting was a 0.43 tok/s baseline inside an MTP benchmark, and my
first explanation was SMB contention from a concurrent model load. That was
wrong. Re-measured with the link quiet it reproduced exactly --- 0.43 and 0.42
tok/s, 1063s per config --- and then reproduced a third time through a
completely different harness (the table above). Reproducible is not
contention.

## What is NOT affected

The MoE VQ path looks clean at these sizes, measured through the MTP
benchmark harness on the same machine:

- Flash-Next VQ 2.1bpw: 18.85 tok/s baseline
- Qwen3.5-397B-A17B VQ 2.2bpw: 17.59 tok/s baseline

So this is pointing at the **dense** VQ read path (`vq_dense.py` /
`dense_shim.py`), not at VQ as a technique. A wider scope run --- the 4.5 and
4.8bpw dense 27B rungs, plus an MoE rung through the SAME harness for an
apples-to-apples control --- is queued; until it reports, the only artifact
directly measured is 27B-VQ-3.9bpw.

## Why this matters

The dense 27B rungs are released artifacts. If they decode at half a token
per second on a machine that runs the stock model at 17, then anyone who
downloaded one has an unusable model, and no quality benchmark we have
published for them reflects what a user experiences.

**Nothing has been changed, unpublished, or announced.** This is a finding,
and the decision about what to do with it is Noah's.

## Suggested first steps

1. Finish the scope run: is it every dense rung, or one bad artifact?
2. Profile a single forward on the VQ artifact. A 13x prefill penalty
   alongside a 39x decode penalty suggests per-call reconstruction work that
   does not amortise over sequence length --- which would point at the
   dequantisation path being re-run per call rather than at the kernel.
3. Compare against the dense VQ artifact's own construction-time numbers, if
   any were recorded; the fit/pack stages may have measured something that
   never got re-checked after assembly.
