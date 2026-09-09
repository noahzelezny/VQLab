# Morning report — 2026-09-09

## The night, in one paragraph
Both overnight swarms initially HUNG for 85 minutes producing zero tokens
(F27: the serving layer wedged; a lone 24k-token prompt timed out). Recovery:
full exo restart, one 27B per node placed EXPLICITLY via POST /instance with
nodeToRunner (F28: the Studio's psutil-fallback memory metric lies, which is
why auto-placement stacks one node), relaunch at concurrency 3 behind a
token-liveness gate. After that: clean. Both banks completed with
checkpoints, and every harness fix earned its keep — per-proposer landing,
resume, and salvage (three non-submitters left 40-57k chars of prose EACH
instead of nothing; contrast the 75k tokens vaporised on 09-08).

## The haul: 27 proposals, and the mission steering worked
Bank A (original frames): 11. Bank B (occupancy/shape/scheduling/numerics/
measurement): 16. Mechanical screen: 25/27 survive (the screen only catches
geometry + dead-path errors — anchor-verify before building ANYTHING).

Nearly all 27 are MEASUREMENTS, clustered on exactly the open mechanisms:

| open question | proposals aimed at it |
|---|---|
| WHY RTILE=64 flips sign on geometry (F25) | a3 a5 a7 · b2 b8 b12 b14 |
| WHY the NSUB=80 fused null (28.4% of work) | a4 a8 · b4 b5 b10 |
| the 8.4% tail-tile waste (F24) | a1 a6 a9 · b11 |
| decode's fixed cost inside the forward (F23) | a0 a10 · b9 b15 |
| occupancy mechanism behind CB_DEV 1.76x | b0 b1 |

Notable singles, unverified: a0 resurrects a DORMANT VQ_DECODE_VEC kernel
flag for the decode path (check it exists before excitement); b14 wants to
decompose the 1.76x into CB_DEV-vs-RTILE components — pointed, since F25
showed the two flags interact; b4's "gate-identity A/B" forces the NSUB=80
shape through gemmseg vs fused vs legacy under one routing.

## Suggested morning order
1. Pick ONE RTILE-flip experiment (b8 and a7 look cheapest: env-flag NGRP
   sweeps) and ONE NSUB-null experiment (b4). Run on the out-of-exo harness
   (probe_moe_prefill_attrib.py pattern — no placements, per-process env).
2. Read the three salvage blobs in swarm4a/ideas.partial.json (fusion,
   batching, host) — 137k chars of 27B reasoning that never got structured.
3. The wedge (F27) is UNREPRODUCED and unexplained: 10 concurrent 24k
   prefills is the suspect. One controlled blast test would settle it.
4. Publish gate (19 artifacts) still awaits Noah.

Everything above is claims-by-swarm until anchor-checked. The night's rule
held: nothing was built on any of it.
