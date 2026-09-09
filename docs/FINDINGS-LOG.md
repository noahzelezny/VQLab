# FINDINGS LOG — the tended record

One entry per finding: the NUMBER, how it was measured, what it changed.
Newest first. This file exists because no context window survives the arc —
write here the moment a number lands, not at the end of the night.
Rules: numbers only if measured; every claim carries its method; corrections
edit the entry in place and say CORRECTED. Keep it readable in one sitting.

---

## 2026-09-08 (one day, in order of discovery)

**F26 · CB_DEV at d4-K512 is a 0.83x REGRESSION, not a tie.**
GLM-2.7, same 2-node placement both arms, 3 reps: threadgroup 251 tok/s,
forced CB_DEV 209. The old "0.997x tie" row was uncontrolled. The 16 KB
threshold is now bracketed from both sides: 4 KB regresses, 16 KB is 1.76x.

**F25 · RTILE=64 flips sign on GEOMETRY: 0.75x on d4-K2048.**
35B-3.4 single-box, out-of-exo harness, 3 interleaved reps: 1834 vs 1379
tok/s, spreads ≤1.2%. Same box where d8-K16384 (Flash-2.1) measured 1.23x.
RTILE=64 requires the CB_DEV budget, so it was UNREACHABLE on d4-K2048 until
today's F21 made ~447 modules eligible — first measurement says regression.
Ship default 32 stands; any future 64 must gate per geometry. Mechanism
unmeasured (NGRP amortization is the candidate).

**F24 · gemmseg tail-tile waste is 8.4%, and sorting cannot fix it.**
Arithmetic on the tile builder: (expert, row_start, nrows) tiles, one partial
tail per expert; 8,304 idle slots of 98,304 at flagship routing, identical at
RTILE 32/64. Only phase 2/3 pay it (phase-1 decode is per-OUT-row). The
count-sort below the gemmseg return is NOT stranded value — gemmseg never had
the pad-to-chunk-max problem the sort solved.

**F23 · Decode is 99.83% inside mlx-lm's forward.**
DEBUG step log (batch_generate.py:482), 397B-2.2, 2000 tokens: overhead
0.06-0.07 ms flat, next 41-45 ms. Kills the per-token-collective hypothesis
all four decode workers converged on (their unanimous "measure first" is why
it cost one run). Also: the documented 3.4x length collapse did NOT reproduce
(1.087x) — GLM-specific or since-fixed, unresolved. Excluded for decode, all
measured: bandwidth (1-5% of 819 GB/s), ring (1.11x, F19), exo loop (0.17%).

**F22 · Decode bandwidth table.** Flash-2.1 9.6 GB/s effective, 397B-2.2
40.2, 397B-3.1 43.6, 35B-A3B 22.5 — against 819 peak. Flash and 35B move the
same bytes/token (0.58 vs 0.54 GB) and differ 2.3x → large fixed cost inside
the forward. Within the 397B family time tracks bytes; across architectures
it does not.

**F21 · CB_DEV arm at cb ≥ 16 KB: 1.76x on 397B-3.1** (38.0 → 21.6 s, 9k
prefill), 1.43x end-to-end on 35B-3.4. The day's largest win. Came from the
one round-1 proposal that asked for a MEASUREMENT.

**F20 · EXO_MLX_MEM_LIMIT_GB=82 left on the M4 alone collapsed 397B-3.1
decode to 5.9 tok/s; removing it: 16.0 (2.7x), prefill 282→324.** Found by
reading the runner's live env (`ps eww`). ring-env.sh itself had blessed the
cap as "legitimately per-node" — the guard's own doc hid the bug. The cap's
original cause (legacy prefill materialization spike) was removed by gemmseg
a day before. ring-env.sh now unsets it on every node.

**F19 · The ring costs 1.11x on decode, not ~2x.** Same model (Flash-2.1),
1 box vs 2 nodes: 18.5 vs 16.6 tok/s. Corrects an inference drawn from
comparing different architectures across different boxes.

**F18 · The M4 reaches the M3 through the ThunderBay (TB3, 40 Gb/s), not a
direct cable.** system_profiler chain: Studio Bus 2 → ThunderBay Flex 8 →
Mac16,5. Four 120 Gb/s ports empty. Not the decode bug (8 KB/token), but the
inference hop shares a bus with model storage. Standing config note.

**F17 · Swarm reliability: anchors good, conclusions unreliable, and the
mechanical screen cannot fix it.** Round-2 audit: 15/16 proposals dead, 6/8
line anchors EXACT. Screen catches 2/15 (geometry + dead-path checks);
premise-checking by substring failed both directions. Four more proposals
died to code-reading during round 3 (status-quo proposal, coalescing false
premise, collective hypothesis, count-sort relevance). Measurement proposals
are 3-for-3 (F21, F23, F26); design proposals 0-for-many. Point swarms at
measurements.

**F16 · Harness defects found+fixed tonight** (each was invisible; warm_first
had zero test coverage): staged warm worker (−800s/run); checkpoint only at
phase end; stream discarded beyond 160 chars (~75k tokens lost — three Flash
workers' full analyses); no total_timeout; fusion frame hollowed by its own
mission; a seed passing tests against a fake more permissive than the real
client. Fixes: be84043a, f22a95e1, af0145d7 (resume-not-kill, salvage,
--resume, 3h cap) — all mutation-verified.

---

### Standing scoreboard (prefill, 9k, vs affine)
397B: VQ at **87%** of affine (338 vs 411 tok/s, F-prior). Gap left: ~13%.
Open mechanisms, ranked: NSUB=80 null (28.4% of work, zero gain);
RTILE geometry flip (F25); tail-tile 8.4% (F24, structural).
Decode: bounded inside the forward (F23) — per-layer dispatch / template
instantiation are the unprobed suspects.
