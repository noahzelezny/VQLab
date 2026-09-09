# Swarm round 2: anchor audit before refutation (2026-09-08)

16 proposals from 4 frames (2 more proposers never finished — 1h+ with no
tokens, serialized behind Scout's own per-endpoint gen-lock). Refutation was
NOT run. Instead every proposal was checked mechanically against the exact
file the swarm was handed: the bundled `model.py` of 397B-2.2 (4107 lines).

The result reorders what the swarm is good at.

## Anchors are GOOD. Reasoning about the anchored code is not.

6 of 8 distinct line anchors land exactly on the named construct:

    _EXPERT_SIMD_MAX_N          L2404  EXACT
    _PACK_FETCH                 L592   EXACT
    _SRC_FUSED_PACKED           L696   EXACT (the dot() line it quotes)
    _EXPERT_ROWS_TG_D8_PACKED   L2354  EXACT
    _default_decode_chunk       L86    EXACT
    _SRC_GEMMSEG phase 1        ~3064  region correct
    cb_bytes rule               L3495  off by ~4 (right clause)
    _GEMMSEG_RTILE              L3467  WRONG (lands on `else codebook`)

So the failure mode is NOT hallucinated locations. It is reading real code
and drawing a conclusion the code contradicts, or that a document already
in the context window contradicts.

## Verdicts

| # | frame | proposal | verdict |
|---|---|---|---|
| — | fusion | all 4 | **DEAD** — premise "the gate is `D == 2`" is false; gate is `D in (2, 4, 8)`. d4 and d8 have been fused for weeks. Its 4th ("remove the opt-in flag") is also false: `_FUSED_GEMM` defaults to `"2"`, i.e. ON |
| 3 | bandwidth | CB_DEV threshold 16KB -> 8KB | **DEAD** — already measured. CBDEV-ARM-2026-09-08 row 3: gemma-26b d2-K2048, cb 8 KB, 0.988x, a tie. That doc was IN its context and states "8 KB ties". Its named beneficiary (d4-K1024, "Flash-3.2 has 126 such modules") does not exist: Flash-3.2 is (2048,4) x254 + (256,2) x18. Fleet-wide there are 91 modules at 8 KB, all d2-K2048, 90 of them gemma-26b |
| 1, 5 | bandwidth, compute | VQ_CODE -> lookup table (same idea, twice) | **DEAD** — `BITS` is a BAKED template constant at every dispatch site (`("BITS", pack_bits)`). `VQ_W/VQ_SH/VQ_OFF` are integer ops on a loop index with compile-time BITS; the compiler folds them on unroll. A runtime LUT ADDS a memory dependency to replace ALU that costs nothing |
| 4 | compute | simdgroup matrix instead of scalar dots | **DEAD, and already run** — gemmseg v2 phase 3 IS `simdgroup_half8x8` (L3286-3298). The kernel comment records that v1's scalar version measured **34% SLOWER** and is why v2 exists. Proposal moves toward what shipped. It also miscredits: the 1.29x is the gemmseg d8 arm, not `_SRC_FUSED_D8_SIMD` |
| 0 | bandwidth | simdgroup-per-row extended to prefill | **DEAD** — d8 prefill already goes through gemmseg, which already uses simdgroup matmul. Also self-refuting: its own risk field reports the full forward regressed 13.6% |
| 7 | compute | vectorize scale application | **OFF-PATH** — targets `_SRC_FUSED_PACKED` (small-N decode). gemmseg already pre-scales at half into threadgroup. Admits it breaks reduction order |
| 8 | batching | batch-aware `_default_decode_chunk` | **DEAD PATH** — `_DECODE_CHUNK` is used ONLY at L3663-3664, downstream of the early return at L3639. When gemmseg fits, `_prefill` returns before reaching it. The fleet is 100% fused, so this loop does not execute |
| 11 | batching | cache decoded weights across chunks | **DEAD PATH** — same early return. gemmseg does not materialize weights at all; that is the point of it |
| 9 | batching | raise rows/threadgroup for d8 at batch>1 | **LIVE but off-mission** — L2582 is `_fused_resolve`, the small-N decode path. Mission was prefill |
| 2, 10 | bandwidth, batching | topology-aware RTILE (64 box / 32 ring) | **NOT A PROPOSAL** — this is our own measurement handed back. RTILE-2026-09-08 already records 1.23x box / 0.82x ring, and that the ring number has uncontrolled rank->box assignment. Still the right open question; the swarm added nothing to it |
| 6 | compute | restructure gemmseg phase-1 decode into batch-load then batch-extract | **SURVIVES** — live path, correct anchor, genuinely unmeasured |

## What survives

**One.** Proposal 6: separate the memory ops from the compute in gemmseg's
phase-1 decode loop so codebook loads pipeline behind extraction, instead of
the current load-extract-scale-store per code. It is on the hot path, its
anchor is real, and nothing we have measured speaks to it either way.

Everything else is dead, already measured, on a dead path, or is our own
prior work returned to us.

## The cost, stated plainly

Round 2 ran ~2h of 397B generation across 4 completed proposers plus 2 that
produced nothing, and yielded ONE idea worth testing. Round 1's single win
came from the one proposal that asked for a measurement rather than a design
change — the same shape as the survivor here is NOT (6 is a design change).

The audit that produced this table cost ~5 minutes of grep and zero GPU.
It should run BEFORE refutation, always: a refuter spends 45-52k tokens
arguing about feasibility, and 10 of these 12 die to `grep -n`.

## Consequence for the harness

Add a mechanical pre-refutation gate: for each proposal, confirm (a) the
anchor resolves, (b) the geometry it names exists in the fleet, (c) the
code path it targets is reachable, (d) no document already in its context
reports the measurement. Only survivors go to refuters — which is also the
argument for putting refutation on a smaller/faster model, since what is
left for it to do is narrower than it looks.
