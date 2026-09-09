# FINDINGS LOG — the tended record

## READ THIS FIRST — the load-bearing facts (2026-09-09)

If you read nothing else in this file, read this block. Everything here is
MEASURED, and several items overturned a confidently-documented claim.

**THE CEILING (F39, REFINED BY F41).** Deleting gemmseg's entire NGRP loop —
phases 1, 2 and 3 — makes prefill 1.49x faster. So the kernel BODY is ~30% of
prefill and an infinitely fast kernel is worth ~1.49x. But deleting the WHOLE
VQ module (F41) gives **1.76-1.78x**: the module is **44% of prefill**, and the
extra **14 points are host, dispatch, gather/scatter and cast** — reachable
from Python, no Metal. That 14% is about the size of the entire remaining
VQ-vs-affine gap. Of it, the unsorted-path argsort/scatter is only 1.2%
(measured); the rest is the per-call numpy tile build, the mx.array uploads and
dispatch, and is UNSEPARATED. Four swarm rounds and 60+ proposals aimed only at
the 30%. **Attention, router and base arch (~56%) still have never been
examined.**

**WHAT ACTUALLY SHIPS AND WORKS**
* CB_DEV at cb >= 16 KB: **1.46x**, verified in two harnesses (F21/F32).
  Mechanism PROVEN = **occupancy**, not codebook locality (F36): injecting
  dead threadgroup bytes with cb untouched reproduces it to three decimals
  (1.464x vs 1.463x). Budget curve: 12 KB -> 2065 tok/s, 20 KB -> 1864,
  28 KB -> 1411.
* Ragged-NSUB relaxation: **1.34x** on Flash-2.1 (F31), NOT the "null" it was
  documented as. The original benchmark ran 7 minutes before the bundle it was
  testing got the change.
* d8 fused arm: ~1.3x, promoted and holding.

**WHAT IS FALSE — do not rediscover**
* RTILE=64 is SLOWER everywhere: 0.75x on d4-K2048, 0.87-0.97x on Flash-2.1,
  across two boxes and two harnesses (F25c/F33). The documented 1.23x and the
  "topology/geometry flips the sign" story do not reproduce. Doc superseded.
* There is no NSUB=80 mystery (F31). It is 1.34x.
* gemmseg tail-tile waste: 8.4% of SLOTS but only **~1.1% of wall time**
  (F37). The whole cluster of tail-tile ideas is dead.
* xsrc scatter penalty: **1.8% of prefill** (F40).
* Unreachable on a 100%-fused fleet (below the gemmseg early return):
  `_decode_chunk`, `_DECODE_CHUNK`, `VQ_DECODE_VEC`, `VQ_MOE_EXACT_GEMM`.
* NSG (simdgroups/threadgroup) is pinned at OTILE/8 = 4 — `sg` indexes the
  output tile. Not a knob.
* `G` is baked into each artifact's scales layout, not an env knob.

**DECODE IS OPEN (F23).** 99.83% of a decode step is inside mlx-lm's forward.
Excluded by measurement: collectives (0.17%), the ring (1.11x), weight
bandwidth (1-5% of 819 GB/s). Flash-2.1 and 35B-A3B move nearly identical
bytes/token and differ 2.3x. Unexplained.

**PUBLISH STATE (arc6, 2026-09-09 15:20).** All 20 local artifacts are
spliced with the frozen release runtime (d8 + CB_DEV + ragged-NSUB + routing
memo) and PASS `check-bundle`; `.pre-arc6` backups sit beside every model.py.
Full `check-release` (load + strict smoke) PASSES on the 13 single-box rungs:
27B x3, 35B x4, gemma x2, Flash 2.1/3.2 (Flash gates must run in the exo
conda env -- the bundles import mlx_vlm.models.qwen4_exp, which vqlab's venv
lacks; that is a harness-env fact, not an artifact defect -- Flash has always
required mlx-vlm). NOT yet smoked: Flash 4.4/5.5, 397B x4, GLM x3 -- they
exceed the M3's 96 GB; 397B-2.2 fits the M4, the rest need the two-node path
arc5 used. The Hub still carries the pre-arc6 runtime everywhere: NOTHING has
been pushed. Push runbook: docs/PUSH-RUNBOOK.md, gated on Noah.

**METHOD RULES THAT EARNED THEIR PLACE TODAY** (each caught a wrong result)
1. **One harness.** Never compare an exo number to a local-probe number. That
   error produced the phantom RTILE flip and three wrong decode causes.
2. **Verify the change is in the RUNNING PROCESS**, not the file you edited.
   `ps eww` the runner. Caught: a ring-env override that would have measured
   RTILE=32 twice; a bundle benchmarked before it was rewritten.
3. **Push an added resource past a hard limit and confirm the expected
   failure.** A pad sweep produced a beautiful flat refutation of occupancy —
   invalid, because the compiler had deleted the never-read array. The
   over-cap case must FAIL or your instrument is not installed.
4. **Attribution beats argument.** Every finding today came from deleting a
   component and timing the difference. 60+ swarm proposals, ~0 adopted; the
   one implemented was bit-exact and 10% slower.

---


One entry per finding: the NUMBER, how it was measured, what it changed.
Newest first. This file exists because no context window survives the arc —
write here the moment a number lands, not at the end of the night.
Rules: numbers only if measured; every claim carries its method; corrections
edit the entry in place and say CORRECTED. Keep it readable in one sitting.

---

## 2026-09-09 (morning, running the overnight proposals)

**F40 · The overnight swarm is finished: 14 of 15 proposals dead, 1 survivor,
5 findings — and every finding came from ATTRIBUTION, not from a proposal.**

Final batch:
* **a2** (contiguous xsrc pre-gather) — DEAD, measured. Swapping the
  `srcrows[]` indirection for a contiguous index (same bytes, same
  instructions, only the address pattern) gives 2050.3 -> 2087.8 tok/s. The
  ENTIRE scatter penalty is **1.8% of prefill**, and a2 proposes adding a
  host-side gather pass to reclaim it.
* **b15** — DEAD twice: anchors line 3666, below the gemmseg return at 3639
  (unreachable), and premised on the NSUB=80 null.
* **b6, b7** — DEAD. Both state their decision criterion as "if B's null
  closes". F31 showed that null is a 1.34x WIN, so there is nothing to close.
* **a10** — DEAD as written. Its named prime suspect, `VQPLEEmbedding`'s
  per-shard micro-dispatch chain, does not exist in either shipping model
  (`vq_ple: no` on 35B-3.4 and 397B-2.2); it is only in the NO-SHIP gemma PLE
  rung. Its general ask (count dispatches per token) survives inside b9.

### Final scorecard, 15 live proposals
    DEAD, unreachable code        a0, b13, b15
    DEAD, structurally impossible b3
    DEAD, measured too small      a1 (0.90x), a2 (1.8% ceiling), a6, a9, b11
    DEAD, falsified premise       b6, b7, b15, a10
    PRODUCED A FINDING            b1 -> F34, b0 -> F36
    STILL LIVE                    b9 (per-module decode breakdown)

**Five findings came out of this: F34, F36, F37, F38, F39.** Not one came
from implementing a proposal as designed. Every one came from ATTRIBUTION
experiments — deleting a component and timing the difference — that I built to
evaluate proposals rather than from the proposals themselves. The single
implemented proposal (a1) was bit-exact and 10% slower.

The pattern across four rounds is now unambiguous and worth acting on: swarms
locate real code and reliably draw wrong conclusions from it, while the cheap
attribution probe answers the question the proposal was arguing about. The
next round should ask for MEASUREMENTS TO TAKE, not changes to make — and
should be pointed at the 67% of prefill that is NOT gemmseg (F39), which no
proposal in four rounds has touched.


**F39 · The ENTIRE gemmseg group loop is 33% of prefill. Deleting all three
phases buys only 1.49x.** And this corrects F38's framing.

    baseline                                   2025.5 tok/s
    NGRP loop forced to zero iterations        3015.5 tok/s   1.49x

So phases 1+2+3 together = 1 - 2025.5/3015.5 = **32.8% of prefill wall time**.
F38's separate arms summed to 33.7% (10.6 + 9.6 + 13.5) — an independent
cross-check agreeing within a point, which validates both the arms and this.

**F38 said "two-thirds is unattributed" and implied a hole INSIDE gemmseg.
Wrong framing.** The percentages are of TOTAL PREFILL, so the other ~67% is
attention, the router, the non-MoE layers, write-back, per-dispatch overhead
and host work — i.e. the rest of the forward pass, exactly where it should be.
There is no mystery residual in the kernel; the three phases ARE the loop and
they add up.

**THE BOUND THAT MATTERS FOR PARITY.** VQ prefill sits at 87% of affine
(AFFINE-BASELINE-397B). If gemmseg's whole body is 33% of prefill, then
making the VQ kernel INFINITELY FAST caps out at 1.49x, and any realistic
kernel win is a fraction of that. The remaining ~13% gap to affine cannot be
closed from inside gemmseg alone — a large part of it lives in the other 67%,
which no proposal in four swarm rounds has targeted.

Consequence for the survivors: a2 (contiguous xsrc pre-gather) attacks the
9.6% x-gather, b6/b7 attack phase-1/2 internals. Their combined ceiling is
~20% of prefill and each would claim only a slice. Worth knowing before
building any of them.


**F38 · gemmseg's three phase bodies are ~34% of TOTAL PREFILL (not of gemmseg).
CORRECTED by F39 — see below; the residual is other model work, not a hole
inside the kernel.** (the per-phase profile proposals a4/b5/b15 asked for)
35B-3.4, 9k, step 4096, each arm deletes one thing and produces deliberately
WRONG output; timing only:

    baseline                        2046.0 tok/s
    skip phase-1 codebook+scale     2289.1      => 10.6% of runtime
    skip phase-2 x gather           2262.3      =>  9.6%
    skip 3/4 phase-3 MACs (F37)     2303.5      => 13.5% (extrapolated to 4/4)
                                                  ------
                                                  ~33.7% accounted
                                                  ~66%   UNATTRIBUTED

CAVEAT ON METHOD, stated because it bounds the claim: the skip arms are
PARTIAL. "skip phase-1" removes the codebook gather and the scale multiply but
leaves `VQ_FETCH` (the packed bit-extraction) and the scale LOAD in place;
"skip phase-2" removes the strided device read but keeps the xt store. So the
unattributed ~66% contains VQ_FETCH, the stores, barriers, write-back and
per-dispatch overhead — it is a residual, not a measured single cause.

**It independently corroborates F36.** The codebook gather that CB_DEV
relocates is only 10.6% of runtime, and CB_DEV is worth 1.46x. Relocating a
10.6% component cannot produce a 46% gain by making that component faster, so
the win must come from something else — which is precisely what the dead-bytes
pad experiment proved it to be (occupancy).

**Consequence for the remaining proposals.** a2 (contiguous xsrc pre-gather),
b6 (device-x) and b7 (lane-span restructure) all target phase-1/2 internals,
i.e. a combined 20.2% ceiling, and each would claim only a fraction of that.
The 66% residual is a bigger target than everything the swarm proposed, and
nobody proposed it — the same shape as F23, where decode's cost turned out to
be 99.83% inside a forward nobody had opened.


**F37 · The 8.4% tail-tile waste is worth at most ~1.1% of wall time. The
whole tail-tile cluster (a1/a6/a9/b11) is dead.**
`nrow` guards the x-gather (3262-3278) and the write-back (3327-3336) but NOT
phase 3, so idle rows do run real simdgroup MACs on zeros. a1 guards the
phase-3 token blocks by nrow — implemented, and it is BIT-IDENTICAL
(logits checksum -7707798016.000000 both arms) and 10% SLOWER
(2050.2 -> 1835.1 tok/s).

Attribution run to explain that, dropping 3 of 4 token blocks unconditionally
(output deliberately wrong, timing only):

    all 4 blocks      2069.1 tok/s
    1 of 4 blocks     2303.5 tok/s     1.11x

So removing 75% of phase-3 MAC work buys 11% => phase 3 is ~13.5% of runtime.
The 8.4% idle SLOTS are 8.4% OF THAT: a ceiling of **~1.1% of wall time**.
a1 paid a 10% branch to chase 1.1%, which is why it lost.

Verdicts, all four from measurement or arithmetic rather than opinion:
* a1 — correct, 0.90x. The opportunity is real and an order of magnitude
  smaller than the mechanism needed to claim it.
* a6 (straddle) — needs a 2nd weight tile in threadgroup, +4-8 KB. F36's
  measured curve prices that at 1.11-1.25x. Costs more than it recovers.
* a9 (pad rows to an RTILE multiple) — replaces zero rows with duplicated
  real ones: same MAC cost, PLUS gathers that nrow currently skips. Strictly
  worse.
* b11 (gate tail work on a tmeta histogram) — gating machinery for a 1.1%
  ceiling. Not worth the dispatch.

F24 stands as arithmetic and is hereby demoted in importance: 8.4% of token
SLOTS is a COVERAGE statistic, not a speed opportunity — the same distinction
D8-BENCH drew about "100% fused". Four of the fifteen overnight proposals were
aimed at it.

Also killed this round: **b3** (raise simdgroups/threadgroup 4 -> 5) is
structurally impossible, not merely untested. `sg` indexes the output tile
(`simdgroup_load(B, &wtT[k8*8][(int)sg*8], 32)`), so NSG is pinned at
OTILE/8 = 4; NSG=5 reads column 32 of a 32-wide array. Raising it needs
OTILE=40, which breaks the 32x32 simdgroup_half8x8 tiling the kernel is built
on.


**F36 · OCCUPANCY IS THE MECHANISM — proven causally, prediction hit within
1%.** (proposal b0, with the instrument fixed per F35)
Dead threadgroup bytes injected into _SRC_GEMMSEG2, `cb` NEVER touched,
CB_DEV arm throughout so the codebook is on device in every cell:

    pad  0 KB -> tg 12 KB   2065.1 tok/s
    pad  8 KB -> tg 20 KB   1863.6
    pad 16 KB -> tg 28 KB   1410.6      (pre-registered prediction: ~1400)

Now against F32's CB_DEV arms, which moved `cb` instead of adding dead bytes:

    pad experiment   2065.1 / 1410.6 = 1.464x     cb never changed
    CB_DEV   (F32)   2027.7 / 1385.4 = 1.463x     cb moved to device

Agreement to three decimals — and the ABSOLUTE values match too: pad=16 KB
puts the total at 28 KB, threadgroup-arm d4-K2048 sits at 28672 B, and they
measure 1410.6 vs 1385.4, 1.8% apart. Same budget, same speed.

**The whole 1.46x CB_DEV win is freeing 16 KB of threadgroup memory.** Not
device-vs-threadgroup latency, not codebook caching, not L2 residency. Sixteen
kilobytes that nothing ever reads reproduce the entire effect. F34's
correlation is now a cause, and the source comment's "occupancy pressure at
the budget edge is a hypothesis, not a finding" can be retired.

CONSEQUENCE FOR THE ARM RULE. `cb_dev = ... or _cb_bytes >= 16384` is written
in terms of CODEBOOK SIZE, but the physics is TOTAL THREADGROUP BUDGET. They
coincide today only because wtT+xt+ybuf is a constant 12 KB at RTILE=32, so
cb >= 16 KB is exactly total >= 28 KB. Any future change to RTILE, OTILE or
ybuf breaks that coincidence silently and the rule stops tracking the thing
that matters. The rule should be expressed as a budget threshold. NOT changed
today — it is behaviourally identical at every shipped geometry, and this is
the wrong hour to touch the arm that 447 modules depend on.

Method note: the pre-registered prediction is what makes this convincing. ~1400
was computed from the CB_DEV ratio BEFORE the run, published in the turn that
launched it, and the measurement returned 1410.6.


**F35 · The occupancy discriminator (proposal b0) ran, produced a clean flat
result, and the result is INVALID — the compiler deleted the instrument.**
Method: inject `threadgroup half _pad[N]`, written at a runtime index and
never read, into _SRC_GEMMSEG2 via the loaded module's globals; sweep N. This
varies TOTAL threadgroup budget without touching `cb`, which is the one thing
F34's four points cannot separate.

    pad  0 KB (tg 12 KB)  2022.4 tok/s
    pad  4 KB (tg 16 KB)  2041.2
    pad  8 KB (tg 20 KB)  2045.8
    pad 12 KB (tg 24 KB)  2056.1
    pad 16 KB (tg 28 KB)  2075.1     <- prediction was ~1400 if occupancy-bound

Flat, even slightly rising. That reads as a clean refutation of occupancy —
and it is worth nothing, because the VALIDITY CHECK failed: pad=24 KB puts the
total at 36 KB, over the 32768 B cap, and it LOADED AND RAN at 1975 tok/s
instead of raising E134. A threadgroup array that is written and never read is
dead-code-eliminated by the Metal compiler, so every point in the sweep
compiled the SAME kernel. The experiment measured nothing.

F34's monotonic correlation is therefore still just a correlation: neither
confirmed nor refuted. Occupancy remains the leading hypothesis and remains
UNPROVEN.

**The check that caught it is the reusable part**: for any experiment that
adds a resource, push the resource PAST a known hard limit and confirm the
failure you expect. If it does not fail, your instrument is not installed.
That is now three saves today from the same discipline — the ring-env override
that would have measured RTILE=32 twice (F33), the ragged bundle whose ADMITTED
arm ran refusing code (F31), and this.

Retry in flight with the array READ behind a never-true guard
(`if ((float)_pad[...] > 1.0e4f) ybuf[0][0] += 1.0f;`) plus a barrier, which
the compiler cannot prove dead while leaving output bit-identical. The over-cap
case MUST fail before any timing from it is believed.


**F34 · The CB_DEV mechanism is occupancy pressure, and it is MONOTONIC in
threadgroup-budget utilisation.** (proposal b1, static audit, zero GPU)
CORRECTED WITHIN THE HOUR: the first pass omitted `ybuf[RTILE][32]` (float,
4 KB @ RTILE=32). v2 declares FOUR threadgroup arrays, not three:
cb[MAX_K] (half2 @ d2 = K*4 B, half4 @ d4 = K*8 B) + wtT[G][32] + xt[RTILE][G]
+ ybuf[RTILE][32]. Corrected against the 32768 B cap:

    geometry    cb     wtT    xt   ybuf   TOTAL  %cap  spare  CB_DEV effect
    d4-K512    4096   4096  4096   4096   16384   50%  16384  0.83x REGRESSION (F26)
    d2-K1024   4096   4096  4096   4096   16384   50%  16384  1.047x
    d2-K2048   8192   4096  4096   4096   20480   62%  12288  0.988x tie
    d4-K2048  16384   4096  4096   4096   28672   88%   4096  1.46x WIN (F32)

The monotonic relationship SURVIVES the correction (50% -> 62% -> 88% rather
than 37/50/75), and the corrected figure now agrees exactly with the source
comment at the arm rule, which says the threadgroup arm at d4-K2048 "leaves
only 4096 B of the 32768 B budget spare". My first pass said 8192 and
contradicted a comment that was right. Read the kernel, not your memory of it.

Monotonic: at 37% of budget moving the codebook to device COSTS you, at 50%
it is a wash, at 75% it is a 1.46x win. That is the "occupancy pressure at the
budget edge" hypothesis the source comment flags as UNMEASURED — now with a
shape and a threshold rather than a guess. Four points, one harness each, so
this is a strong correlation and not yet a proven cause; the clean
discriminator would vary total budget WITHOUT varying cb, which the current
flags cannot do (RTILE is the only xt lever and it is gated on cb_dev).

COROLLARY, and it is a nice one: d4-K2048 is the ONLY fleet geometry that is
threadgroup-LEGAL yet forced to device by the `>= 16384` clause. Everything
larger is budget-forced regardless; everything smaller is left on threadgroup,
where the audit says it belongs. The clause is precisely targeted at the 447
modules where it pays and touches nothing else.

Also killed cheaply this round, both by reachability rather than measurement:
proposal a0 (`VQ_DECODE_VEC`, a genuinely dormant flag) and b13
(`VQ_MOE_EXACT_GEMM`) both gate code below the gemmseg early return at line
3648 — unreachable on a 100%-fused fleet.


**F33 · The exo-vs-local RTILE discrepancy is CLOSED: exo agrees, RTILE=64 is
slower there too. The documented 1.23x does not reproduce in any harness.**
Flash-2.1, M4 single-box, 9k prefill, 3 reps, flag set through ring-env.sh
(the canon) and verified in the live runner's env before each arm:

    exo   RTILE=32  519 tok/s     RTILE=64  493 tok/s     0.95x
    local RTILE=32  472-497       RTILE=64  425-427       0.87x

Same box, both harnesses, same direction. Combined with F25-corrected
(35B-3.4 uniform d4-K2048 at 0.75x, matched harness), RTILE=64 is now slower
in FIVE independent measurements across two boxes, two harnesses and two
artifacts. The 1.23x in RTILE-2026-09-08 is an erroneous measurement, not a
configuration this repo can reach. Shipped default (32) unchanged.

PROCESS NOTE — a near-miss worth keeping. The first attempt set
VQ_MOE_GEMMSEG_RTILE=64 in the M4's `~/.exo/exo-env.sh`. That file is sourced
BEFORE `ring-env.sh`, which assigns RTILE=32 unconditionally by design, so the
edit would have been silently overridden and arm 2 would have measured
RTILE=32 a second time — reported as "no difference". Caught only by reading
the live process env (`ps eww`) before measuring. This is the same
unapplied-change shape as F31 and the CB_DEV `cd` short-circuit. **Verify the
flag in the running process, not in the file you edited.**


**F32 · CB_DEV (F21, the arc's largest win) VERIFIED independently: 1.46x.**
35B-3.4, M3, 9k, step 4096, warm rep discarded, local harness — a different
harness from the one that produced F21's 1.43x:

    CB_DEV=1 (device, shipped)      2027.7 tok/s
    CB_DEV=0 (threadgroup, pre-F21) 1385.4 tok/s     = 1.46x

Within 2% of the original claim. The threshold rule shipped yesterday
(`cb_dev = budget-forced or _cb_bytes >= 16384`, line 3492 of the bundle) is
sound, and the ~447 modules moved to the device arm were moved correctly.

### Verification scoreboard for the arc's three headline numbers
    CB_DEV       claimed 1.43x   ->  VERIFIED 1.46x
    RTILE=64     claimed 1.23x   ->  DOES NOT REPRODUCE (0.75-0.97x, 4 runs)
    ragged NSUB  claimed null    ->  WRONG, it is 1.34x
Two of three documented claims were wrong; the one the fleet default rests on
is right. Both errors were cross-harness or unapplied-change artifacts, not
kernel physics — and both were caught by re-running in ONE controlled harness,
which is now the standing requirement before any number enters this log.


**F31 · The ragged-NSUB relaxation is worth 1.34x on Flash-2.1 — it was
shipped and documented as a NULL, and that null was almost certainly measured
against a runtime that did not yet contain the change.**
Controlled 2-arm, one harness, M3, 9k, prefill_step 4096, warm rep discarded:

    ragged ADMITTED  834.3 / 825.0 tok/s   (spread 1.1%)
    ragged REFUSED   615.3 / 616.3 tok/s   (spread 0.16%)   = 1.34x

Refusal reimposes the pre-relaxation gate by wrapping the LOADED module's
gemmseg_fits; instrumentation confirms it rejects exactly 552 calls, all
`d8 K16384 IN=640 NSUB=80`, and nothing else.

WHY THE ORIGINAL SAID NULL — the arms diverge in a diagnostic way:

    arm        D8-BENCH (exo)   here (local)   agree?
    REFUSED    601 tok/s        615, 616       YES, ~2%
    ADMITTED   598 tok/s        825, 834       NO, 40%

Two different harnesses land on the SAME refused number, which validates
comparing them; they diverge only on the arm whose code changed. And
`model.py` in the artifact was rewritten at 10:36 while D8-BENCH was written
at 10:29 — the doc's ADMITTED arm was very likely running a bundle that still
refused, i.e. it measured the same path twice and correctly reported 0.995x.
Same failure shape as the CB_DEV patch a `cd` silently short-circuited: a
confident null measured on an unapplied change. mtime is circumstantial;
the 1.34x is not.

SCOPE: Flash-2.1 only. The 397B (hidden 4096, moe_inter 1024) has d8 experts
at NSUB=512 and NSUB=128, both multiples of 32 — no ragged modules, nothing
to gain. KERNEL-COVERAGE records Flash-2.1 as the only artifact that was
below 100% fused. The published Flash-2.1 bundle DOES carry the relaxation
(mtime 10:36), so downloaders already get the 1.34x; only the documentation
was wrong.

CONSEQUENCE: D8-BENCH's "NSUB=80 relaxation is a NULL" section, and every
inference drawn from it — including "at in=640/out=2560 the fused kernel is
not faster than legacy", which the overnight mission carried forward as a
headline open mechanism and five proposals were aimed at — rests on that
measurement. The small-NSUB shape is not mysteriously null; it is 1.34x.


**F30 · The NSUB=80 "same kernel, opposite result" framing is WRONG for the
decode path and RIGHT for prefill — and the two got conflated.**
Proposal b4 claimed the two d8 shapes run different kernels, explaining the
null with no new physics. Verified in `_fused_resolve` (line 2557,
`elif simd and IN // G >= 32`), G=64, K=16384 > _D8_TG_MAX_K=1024:

    A  in=2560 out=640  NSUB=320:  IN//G = 40  -> PASSES -> simd kernel
    B  in=640  out=2560 NSUB=80 :  IN//G = 10  -> FAILS  -> device-cb variant

True. But that gate is the SMALL-N DECODE path. The null was measured in
gemmseg PREFILL (D8-BENCH:126-127, ragged REFUSED 11.587 s vs ADMITTED
11.645 s), and gemmseg dispatches exactly one source (`_SRC_GEMMSEG2`,
line 3465) with no shape branching — so there both shapes DO run the same
kernel and b4's mechanism cannot explain the null.
SURVIVING FACT, new and unlogged: on the decode path shape B falls off the
simd kernel onto the device-codebook variant. Worth measuring separately;
NOT an explanation for the prefill null.

**F29 · Proposal a7 is unrunnable: G is not an env knob.**
Its 2x2 sweep varies G (64 vs 256) by flag. Grep of the bundled runtime finds
no env control of G — the only ROWS_TG-style knob is `_WDEC_ROWS_TG`. G is the
scale-group size BAKED INTO each artifact's `scales` tensor layout; changing it
means repacking weights and re-uploading. The sweep axis does not exist.

**F28b · The out-of-exo probe needs the EXO env on the M3, not the repo venv.**
`probe_moe_prefill_attrib.py` on Flash-2.1 dies in the repo `.venv`
(`ModuleNotFoundError: mlx_vlm.models.qwen4_exp`) — the same gate-venv limit
D8-BENCH recorded. Use `/opt/anaconda3/envs/exo/bin/python3.13` (M3) and
`/opt/homebrew/anaconda3/envs/exo/bin/python3.13` (M4). Also: `timeout(1)` does
not exist on the M4, and its absence silently turned six benchmark cells into
"FAILED" that were never run.

---

## 2026-09-09 (overnight)

**F27 · The overnight swarms hung for 85 minutes producing zero tokens, and
nothing noticed.** Both swarm4 runs spawned 5 workers, opened sockets, and
never streamed one token; M4 runner CPU 4.8%; a single 24k-token prompt then
timed out at 280s — the serving layer was wedged. Two holes: (a) launch
verification stopped at "N connections established", which is exactly the
looks-alive trap; liveness = TOKEN COUNTS ADVANCING, nothing less. (b) the
1500s silence timeout did not fire on a never-started stream in 85 min —
harness hole, unfixed, logged for the morning. Prime suspect for the wedge:
10 concurrent ~24k-token cold prefills against two co-located instances
(gen-lock exempted), unproven. Recovery: full exo restart, one 27B per node
placed EXPLICITLY (POST /instance with nodeToRunner), relaunch at lower
concurrency with a token-liveness gate.

**F28 · Placement stacks one node because the Studio's memory metric lies.**
exo on the M3 falls back to psutil ("macmon not found"); it reported 14.9 GB
available while memory_pressure said 61% free (~30 GB reclaimable inactive +
20 GB swapped). filter_cycles_by_memory believes the metric, so every 27 GB
auto-placement excluded the Studio and stacked the M4. The HTTP
place_instance param schema exposes NO node targeting (required_nodes is
internal-only); explicit placement = POST /instance with a full Instance
carrying nodeToRunner. Also corrected: the "both on the Studio" claims
earlier were memory-delta inferences; nodeToRunner is the ground truth and
said M4 both times it was checked.

---

## 2026-09-08 (one day, in order of discovery)

**F26 · CB_DEV at d4-K512 is a 0.83x REGRESSION, not a tie.**
GLM-2.7, same 2-node placement both arms, 3 reps: threadgroup 251 tok/s,
forced CB_DEV 209. The old "0.997x tie" row was uncontrolled. The 16 KB
threshold is now bracketed from both sides: 4 KB regresses, 16 KB is 1.76x.

**F25 · CORRECTED 2026-09-09 — there is NO geometry flip. RTILE=64 is
uniformly slower in every controlled measurement.**
Matched harness (mlx_lm.generate, M3, 9k, warm rep discarded, min of 2):

    35B-3.4  uniform d4-K2048    R32 2002/2067   R64 1515/1558   0.75x
    Flash-2.1 mixed d8           R32  830/ 818   R64  770/ 791   0.93-0.97x

Flash's milder penalty is DILUTION, not physics: only 138 of its 272 modules
are RTILE-eligible (d8-K16384, cb 256 KB, CB_DEV forced); the other 128 are
d8-K256 (cb 4 KB, threadgroup) and silently stay at RTILE=32. A ~25% penalty
applied to ~51% of modules lands at ~0.93x. One mechanism, two dilutions.
Also falsified en route: the prefill-chunk-size hypothesis (RTILE=64 needs
enough rows/expert/dispatch). Swept 2048/4096/8192 — 0.93/0.97/0.97x, sign
never flips.
The ORIGINAL F25 claim ("the sign flips on geometry") came from comparing the
doc's EXO Flash number against my LOCAL 35B number — cross-harness, the same
uncontrolled-comparison error as F19 and the model bake-off. The shipped
default (32) is unchanged and still correct; the reasoning is now simply
"RTILE=64 is slower," with no geometry gate needed.
OPEN: the doc's exo-measured 1.23x on Flash-2.1 has NOT reproduced under any
local harness (3 independent runs, both boxes). Either exo's serving path
interacts with RTILE differently, or that number is an artifact. Requires an
exo-side re-measurement to close.

**F25-original (superseded) · RTILE=64 flips sign on GEOMETRY: 0.75x on d4-K2048.**
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

## F41 (2026-09-09) — the kernel is 30% of prefill, but the VQ MODULE is 44%. The 14% outside the kernel body is addressable without touching Metal.

F39 deleted gemmseg's NGRP loop (phases 1+2+3) and measured 1.49x, which the
READ-THIS-FIRST block turned into "an infinitely fast kernel is worth 1.49x,
the other 67% has never been examined". That arm kept the dispatch, the
per-call numpy tile build, the tmeta/src_rows uploads, the gather/scatter and
the output cast. **Deleting the whole module measures those too**, and the gap
is large.

One harness (`scratchpad/whole_module_delete.py`), 35B-A3B-VQ-3.4bpw, 9000
tokens, prefill_step_size 4096, two independent runs:

| arm | run 1 | run 2 | vs baseline |
|---|---|---|---|
| baseline | 2048.6 | 2036.2 tok/s | 1.00x |
| F39 arm: NGRP loop -> 0 iterations (kernel BODY deleted) | 2930.7 | 3012.5 | **1.43-1.48x** |
| WHOLE module deleted (`__call__` returns zeros) | 3609.7 | 3632.3 | **1.76-1.78x** |

The middle row reproduces F39 in this harness, so the third row is comparable
to it. Wall-time decomposition at 4.42 s baseline:

* kernel body ...................... ~30% of prefill
* VQ module OUTSIDE the kernel body . ~14% of prefill
* everything else (attention, router, base arch, host) ~56%

**The correction to F39's framing.** The VQ path is 44% of prefill, not 33%,
and 14 points of it are host/dispatch/gather/cast -- reachable from Python,
with no Metal involved. That 14% is about the size of the entire remaining
VQ-vs-affine parity gap (~13%).

**What it is NOT (measured, same harness).** Forcing `sorted_indices=True`
deletes both numpy argsorts, the `inv` permutation and the `y[inv]` row-scatter
and buys only **1.2%** (2073.5 vs 2048.6). The unsorted-path overhead the swarm
kept nominating is real but small; the remaining ~12-13% is the per-call numpy
tile build in `_gemmseg_prefill`, the `mx.array` uploads, kernel dispatch, and
the broadcast/cast in `__call__`. Not yet separated.

**INVALID ARM -- do not cite.** Stubbing `np.argsort` to an identity
permutation ran SLOWER than baseline (1984.8 tok/s). It is not an attribution:
handing the kernel unsorted expert order changes its memory access pattern, so
the arm alters the thing it is trying to hold fixed. Recorded because it looks
like a finding and is not.

Four independent swarm frames (bank B `occupancy`, bank A `fusion`, `batching`,
`host`) nominated `_gemmseg_prefill`'s numpy prefix. This arm says their target
region is worth ~12-13%; it does not yet say the prefix is the part.

## F42 (2026-09-09) — splitting F41's 14%: the host prefix is 3.9%, not 12%. The four frames that nominated it were right about the place and wrong about the size.

F41 left ~14% of prefill inside the VQ module but outside the kernel body, and
four independent swarm frames all nominated `_gemmseg_prefill`'s numpy prefix.
Same harness as F41 (35B-A3B-VQ-3.4bpw, 9000 tokens, step 4096).

**Host wall inside `_gemmseg_prefill`** (instrumented, output CORRECT — this is
a wrap-and-accumulate, not a deletion): 360 calls per prefill, **0.171-0.177 s
= 3.8-4.0% of wall**. Treat as an UPPER bound: the timer also catches any queue
backpressure on the `mx.array` uploads.

**Running total of F41's 14%:**

| slice | cost | how measured |
|---|---|---|
| `_gemmseg_prefill` numpy prefix + uploads | 3.9% | wrapped timer |
| argsort + inv + `y[inv]` scatter | 1.2% | `sorted_indices=True` arm (F41) |
| dispatch + GPU-side gather/cast + `__call__` preamble | **~9%** | UNSEPARATED |

So the prefix is real but a third of what the convergence implied, and the
biggest remaining slice is still unattributed.

**A cache is worth ~2.6%, not 3.9%.** The tile build is keyed on the routing
bytes, and **33% of calls repeat within one forward** (240 of 720) — exactly
1 in 3, matching the gate/up/down projections of one MoE layer sharing routing.
A correct memo collapses three identical prefixes into one, so it can recover
two thirds of 3.9%.

**Watch the trap here.** The first version of that probe reported **100%** cache
hits, because the reps replay the same prompt and the cache persisted across
them. It was measuring "the same prompt routes the same way", which no shipped
cache could exploit. Clearing per rep gives the 33% that is actually reachable.

**INVALID ARM — do not cite.** Wrapping `VQSwitchLinear.__call__` in the same
host timer reported **93.5% of wall**, of which 89.5% "preamble/epilogue". That
is not host work: `np.array(idx_flat, copy=False)` forces evaluation of a lazy
MLX array, so the timer blocks on the GPU queue and measures GPU time as host
time. Under MLX laziness a host timer is only meaningful around code that
touches no unevaluated array — which is why the `_gemmseg_prefill` number
(whose input is already a numpy array) stands and this one does not. Second
arm today killed by measuring something other than what it named; see method
rule 1.

## F43 (2026-09-09) — the 9% splits: dispatch is ~2%, the __call__ epilogue/preamble GPU ops are ~4-5%. The full VQ-module ladder, one sitting.

New arm: **null-kernel** — `_gemmseg_prefill` runs its entire numpy prefix
(bincount/nonzero/cumsum, the Python tile loop, the tmeta upload) and then
returns zeros WITHOUT dispatching. Between the existing arms this isolates the
last unattributed slices. Two full ladders, same harness as F41/F42:

| arm | run 1 | run 2 | step means |
|---|---|---|---|
| whole module deleted | 2.48 s | 2.50 s | — |
| null-kernel (host prefix, no dispatch) | 2.88 s | 2.94 s | +~9-10% preamble+host |
| empty loop (dispatch, empty body) | 2.99 s | 3.01 s | +~2% dispatch+write-back |
| baseline | 4.47 s | 4.38 s | +~31-33% kernel body |

**Final decomposition of the VQ module's 44% of prefill:**

* kernel body .......................... ~31-33%
* `__call__` preamble/epilogue GPU ops .. **~4-5%** (largest non-kernel slice)
* `_gemmseg_prefill` host numpy prefix .. ~3.9% (F42)
* kernel dispatch + write-back ......... **~2%** — dispatch is nearly free
* argsort + inv + y[inv] scatter ....... ~1.2% (F41)
* residual / run-to-run ................ ~2%

The dispatch-cost class of proposals (fewer/larger dispatches, batched
launches) is now **dead**: ~2% ceiling. The live target is the ~4-5% of
per-call GPU ops around the kernel — prime suspect is the output
`y.astype(in_dtype).reshape(...)` at the tail (a real fp16->bf16 conversion of
[N, OUT] per module if activations are bf16), exactly salvaged proposal
compute:5. NOT yet attributed further; the broadcast_to preamble may be free
under laziness on the prefill branch (the fused-gather path reads x2 directly
and xf's graph node may never evaluate).

**INVALID ARM — do not cite (third of the day).** "no-out-cast" fed x to the
module pre-cast to fp16 so the tail astype would be an identity. It ran
SLOWER than baseline (1801.6 tok/s): the wrapper ADDS a full-tensor cast of
the [T, IN] hidden state on every call and the module still performs its own
casts. It removed nothing. A deletion arm must delete; wrapping the boundary
added work and measured the addition.

## F44 (2026-09-09) — deleting the tail cast makes prefill 12% SLOWER. The dtype-boundary class is closed for Python-level fixes.

compute:5 (salvaged) asked what the fp16<->bf16 boundary casts cost. Activations
are bf16 (norms/gates bf16, measured), codebook fp16, so the tail
`y.astype(in_dtype)` is a real conversion of [N, OUT] per module. Arm done
properly this time: a symlinked artifact copy whose model.py returns `y` in
fp16 with the astype deleted (nothing added, nothing wrapped).

* baseline ............ 2035.4 tok/s
* tail cast deleted ... **1796.2 tok/s (0.88x)**

And this reproduces the number the INVALID boundary-cast arm produced (1801.6)
— that arm was invalid for its stated purpose but was accidentally measuring
the same thing: fp16 activations meeting bf16 weights downstream. Mixed-dtype
ops promote, and the promotion costs far more than the cast saves. The cast is
PROTECTIVE, not overhead.

Consequences:
* Hoisting/eliding the output cast in Python: **dead** — it makes things worse.
* The remaining ~4-5% preamble/epilogue is dominated by the fp16 boundary
  round-trip (x2 bf16->fp16 in, y fp16->bf16 out, per module call), and the
  only real lever left is a kernel variant with bf16 I/O — a Metal change with
  a ~4-5% ceiling. Parked; below the effort bar while splice+validate is due.

With F41-F44 the VQ module's 44% is fully attributed and every non-kernel
class has a measured verdict: dispatch dead (~2%), argsort/scatter small
(1.2%), host prefix cacheable for ~2.6%, dtype boundary protective. The only
still-unexamined prefill territory is the non-VQ 56%.

## F45 (2026-09-09) — routing memo SHIPPED into the runtime: +1.9% prefill, logits bit-identical.

The one surviving Python win from F42. `_gemmseg_prefill`'s tile build and the
fused-gather branch's argsort trio are memoised on the routing BYTES (exact
memcmp keys, FIFO-bounded at 8). Gate/up/down share a layer's routing, so 2 of
3 rebuilds are eliminated.

Acceptance, one harness, fresh bundle vs the deployed 3.4bpw artifact:
* logits checksum, 4096-token forward: **-3428763136.0 both** (identical)
* prefill: 2046.3 -> **2084.9 tok/s (+1.9%)** (predicted ~2.6% from the 33%
  hit rate x 3.9% prefix; measured a bit under, as usual)

`tests/test_routing_memo.py` pins exact keying (same bytes / different RTILE
or E must MISS), the FIFO bound (mutation-verified), and that a memo hit's
tmeta is bit-identical to an independent rebuild. check-bundle PASSES on a
bundle written from this runtime. Neighbor suites green (47).

## F46 (2026-09-09) — fresh-eyes review (Fable): decode nocast is +3.6% but NOT bit-identical; a10's epitaph is wrong for Flash. Full review: scratchpad/fable_review.md (6 proposals, decode program).

An outside review of the whole runtime + F16-F44 concurred prefill's VQ module
is exhausted and produced six anchored proposals, all decode / non-VQ. Two were
cheap enough to run before the release freeze:

**Decode fp16 round-trip deletion (#5): +3.6%, but numerics change.** The
fused decode kernels are templated on `x.dtype`, so at decode bf16 can flow
straight through and both boundary casts genuinely disappear (F44's "only
lever is a Metal change" was a prefill-only conclusion). Measured on 35B-3.4,
one harness: 53.3 -> 55.2 tok/s decode (+3.6%). BUT the decode-shape logits
checksum moved (-3138743.5 -> -3126615.75): with T=bf16 the kernel's rounding
differs from the fp16-staged path, so the review's "bit-identical for
halfN-staged kernels" prediction is WRONG on d4. A numerics-changing +3.6%
needs the quality gates, not just a checksum — **parked out of this release**,
first item of the post-release decode program.

**a10's epitaph is wrong (#6, verified).** F40 dismissed the PLE
micro-dispatch chain because `vq_ple: no` on 35B-3.4 and 397B-2.2 — but the
Flash-2.1 bundle's config CARRIES `vq_ple` and its model.py instantiates
VQPLEEmbedding. Flash pays ~70-80 tiny PLE dispatches per token that 35B does
not, and Flash is the slow side of the unexplained decode 2.3x. a10 is dead on
the two models it was checked on and LIVE on Flash.

**The decode program (post-release, in order):** (1) the F41 deletion ladder
run AT DECODE on Flash + 35B — note the pre-registered number already in the
source: a 2026-09-02 stub-ablation comment (vq_switch.py ~line 419) measured
the pre-simd-fix VQ expert path at 9.84 ms/token vs stock's 3.09 on Flash;
(2) the fixed-code / no-cb kernel arms — F22's bytes/token accounting is
blind to Flash's dependent codebook-gather chains vs 35B's threadgroup
codebook, which is exactly the shape of the 2.3x; (3) launch census per
decode step (encoder count via Metal capture); (4) attention-deletion arm on
the non-VQ 56% of prefill; (5) ship-or-kill decode nocast after gates;
(6) PLE stub arm on Flash.

## F47 (2026-09-09) — attention is HALF of prefill (linear attention alone 29%), and decode decomposes for the first time: VQ 31% / attention 25% / remainder 37% / +7% launch overlap.

Measurement agent's full report: `scratchpad/fable_measure.md` (probes
`attn_delete.py`, `decode_ladder.py`). 35B-A3B-VQ-3.4bpw, F41 harness for
prefill; decode via stream_generate, 200 tokens, shares within one harness.

**Prefill (baseline 4.31 s / 2089 tok/s).** The arch is HYBRID: 30
GatedDeltaNet (linear-attention) + 10 Qwen3NextAttention layers.

* VQ module -> zeros: 43.2% (reproduces F41's 44% in this process)
* ALL attention -> zeros: **49.2%**, splitting EXACTLY additively into
  GatedDeltaNet **29.0%** + full attention 20.2% (SDPA core 14.6% +
  QKV/RoPE/o_proj 5.6%)
* remainder ~7.6% (norms, router, embedding, lm_head, glue) — a floor, since
  decode showed super-additivity

**The headline: GatedDeltaNet, never on any suspect list in five rounds of
proposals, is the single largest non-VQ prefill consumer at 29%.** Caveat as
measured: zeroing attention perturbs routing downstream, so 49.2% may be a
mild over-credit; the exact additivity of the sub-arms is the consistency
check we have.

**Decode (baseline 17.71 ms/tok / 56.5 tok/s) — first-ever decomposition:**

| slice | ms/tok | share |
|---|---|---|
| VQ module | 5.44 | 30.7% (rep spread makes this a 24-31% band) |
| attention (both kinds) | 4.44 | 25.1% |
| remainder (router, norms, lm_head, sampling, loop) | 6.58 | **37.2%** |
| super-additive interaction (both deleted) | 1.25 | +7.1% |

An infinitely fast VQ module buys decode at most **1.44x** on 35B. The
remainder is the LARGEST slice and has never been examined; its prime suspect
is lm_head (a vocab-width gemm every token). The +7.1% super-additivity is
the classic launch/gap-overlap signature — direct support for the launch-
census hypothesis (F46 program item 3).

Cross-check: an independent manual-step-loop ladder (decode_ladder2.py) gives
shares within a few points (VQ ~29%, attn ~24%, remainder ~44%) — convergent
across two harnesses, never mixed in one comparison.

**Contention note.** A same-afternoon lm_head arm produced a 64 ms/tok
baseline — the Studio GPU was at 100% serving interactive traffic. Numbers
taken while the box serves are garbage; every table above predates that.
lm_head arm queued for an idle window.

**Agent's ranked next steps:** (1) split the decode remainder (lm_head arm,
recorded-routing router arm, launch census); (2) the same decode ladder on
Flash-2.1 — the 2.3x question now has a per-slice baseline to diff against;
(3) probe GatedDeltaNet's 29% before any further gemmseg work.

## F48 (2026-09-09) — the decode 2.3x is NOT the VQ runtime. Flash's VQ module costs the same absolute ms as 35B's; the whole gap is in the non-VQ trunk. Plus: lm_head 2.5%, GDN 29% = honest compute.

The three wrap-up arms, all on the idle M3.

**Flash vs 35B decode, same script, same box, same env** (the exo conda env —
Flash's bundle is a VLM and imports mlx_vlm.models.qwen4_exp, which .venv
lacks; the 35B reference arms were re-run in the same env for comparability):

| | 35B-3.4 | Flash-2.1 |
|---|---|---|
| decode baseline | 18.64 ms/tok | 54.31 ms/tok (2.9x) |
| VQ module (-> zeros) | 5.47 ms (29%) | **6.54 ms (12%)** |
| PLE (-> zeros) | n/a | ~2.7 ms (~5%, 128 shards) |
| non-VQ trunk | 13.2 ms | **~45 ms — 3.4x the 35B's** |

**The unexplained 2.3x (F22/F23) is answered in kind: it is not in the VQ
kernels.** Flash's VQ module is within 1.2x of 35B's in absolute terms; the
entire gap is the non-VQ trunk — the mlx_vlm text path, attention stack, PLE
(~5%), per-layer op count. Suspects for the ~45 ms are the launch census and
the attention arms (Flash's attn-zeros arm still crashes on its VLM cache
classes — the stub seed written for the 35B arch does not transfer; left for
the census pass). The VQ runtime is exonerated at decode on both models.

**lm_head is 2.5% of 35B decode** — the "prime suspect" for the 37% remainder
is a bandwidth blip (19.3 -> 18.8 ms/tok). The remainder is therefore most
likely many-small-ops / launch gaps, consistent with F47's +7.1%
super-additivity. Two invalid arms on the way to this number, both logged
shapes: a class-level stub of QuantizedLinear deleted every attention
projection (lm_head shares the class — patch INSTANCES, not classes), and a
free-standing zeros return made the whole model dead code under laziness
(4.20 ms/tok, "lm_head 78%") — a terminal-node stub MUST depend on its input.

**GDN's 29% of prefill decomposes to honest work:** delta-rule scan kernel
8.0%, depthwise conv ~2%, projections+norms ~19% — and the shard headers show
the five GDN projections are already affine-quantized (U32+scales), so the
19% is quantized-GEMM compute, not an oversight. No cheap runtime lever.

**Attribution campaign closed.** Prefill: VQ 43% (fully decomposed, F41-F45)
+ attention 49% (GDN 29 = honest compute; full attn 20) + ~8% rest. Decode:
VQ ~29-31% / attention ~25% / remainder ~37% (lm_head exonerated; launch-gap
signature). Cross-model: the 2.3x is not ours. Post-release program: launch
census (Flash first), decode-nocast through the quality gates, bf16-I/O
kernel variant (4-5% prefill ceiling).
