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

**PUBLISH STATE — SHIPPED (2026-09-09 17:45).** All 20 repos on the Hub
carry the arc6 runtime, the updated card with the 2026-09-09 changelog entry,
and (11 repos) the MTP sidecar. Every upload went through a clean release
gate WITH a generation smoke: 11 single-box gates on the M3, 9 cluster gates
(--cluster-smoke through a live 2-node exo instance placed per rung and torn
down after; the gate realpath-checks the local rank and hashes the peer's
copies). One refusal en route, root-caused: the M4's 397B-2.2 config.json was
an older serialization (semantically identical) — the peer-hash check caught
it; byte-synced and re-gated. gemma-e4b-PLE published for the first time
since its Metal defect cleared. exo fork pushed through 95ab95c7 (--mtp flag,
recv-dtype + event-log fixes). Ops notes: big-rung loads MUST be from a
node-local copy — tar-over-ssh push per exo-stage-m4.sh (~7 min for 141 GB);
an SMB-symlink load sat at 45% after 30 min. DeepSeek-V4-Flash and
spicyneuron-2.6bit were deleted from the M4 for staging room, both
diff-verified complete on the Thunderbay SSD first.

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

## F49 (2026-09-10, overnight) — the parity gap is the WRAPPER, not the kernel: VQ's kernel body is within ~9% of affine's entire expert path.

The measurement the parity round was built around: the F41 deletion ladder run
on an AFFINE build for the first time. mlx-community/Qwen3.6-35B-A3B-8bit,
same 9k/4096 harness, MoE block -> x*0 (residual keeps the trunk live):

| | affine 8bit | VQ 3.4bpw (F41) |
|---|---|---|
| baseline | 3.43 s (2622 tok/s) | 4.42 s (~2040) |
| expert path deleted | 2.17 s | 2.48 s |
| **expert path cost** | **1.26 s (37%)** | **1.94 s (44%)** |

VQ's expert path is **1.54x** affine's in absolute seconds. Decomposed
(F41-F43): kernel body 1.37 s + wrapper 0.57 s. **The kernel body alone is
within ~9% of affine's whole expert path** — the gemmseg Metal work is
essentially parity-class. The gap is the WRAPPER: dtype round-trip ~0.20 s,
host prefix ~0.17 s (memo recovers part), dispatch ~0.09 s, scatter ~0.05 s.
Closing the wrapper puts VQ at ~93-95% of affine; the last ~8% must come out
of the kernel body (first candidate: the xt leading-dimension padding lifted
from quantized.h — swarm7's one "viable" kernel verdict).

**Dense is already at parity.** The matched dense pair (27B-VQ-4.8 vs
27B-8bit, local, one harness): baselines 388.1 vs 394.7 tok/s = **98.3%**,
and the quantized MLP path itself 14.84 vs 14.09 s = **1.05x**. At 4.8 bits
vs 8. The 13% story was always MoE-only.

**INVALID ARMS (two, recorded).** zeros_like-stub attention on the AFFINE
builds constant-folds the whole graph (0.23-0.27 s "runs"); the VQ builds
never folded because their host numpy ops pin the graph — an asymmetry that
makes stub arms silently invalid on pure-lazy models. And an x*0 stub on a
per-expert-shaped module broke against the router-scores reshape — stub the
x-shaped BLOCK, not the expert GLU. Attention's affine share was therefore
taken from F47's absolute (2.12 s, identical stock trunk modules), not from
a folded arm.

**Runner reliability finding (same night).** Both VQ serving runners (27B-4.8
and Flash-4.4) crashed with `[metal::malloc] Resource limit (499000)
exceeded` under sustained long-output decode (40k-token budget, 2 streams).
Stock models ran the same harness for hours the previous day. 499000 smells
like a Metal BUFFER-COUNT limit — hypothesis: the VQ decode path accumulates
small allocations per step. Reproduce locally watching active buffers; if
real, this is a shippable reliability fix and belongs ahead of any speed
work.

**F49 addendum — leak probe result.** A 2000-token local decode on the 3.4bpw
shows NO byte-level leak: active memory 13.984 -> 14.020 GB (linear KV growth,
~18 KB/token, exactly the cache), allocator cache 0.045 -> 0.196 GB, peak
stable. The shipped runtime's single-stream decode is clean. So the runner
crash is NOT a simple runtime memory leak; suspicion moves to (a) a Metal
buffer-COUNT limit these byte counters cannot see, or (b) exo's serving layer
under repeated large-prompt requests — the per-request prefix-cache
deepcopy+pool is the named suspect in llm_client's own comments. Next
reproduction must go THROUGH exo serving (repeated 25k-prompt requests, two
streams), not local mlx_lm. Daylight item; the artifacts themselves are not
implicated by current evidence.

## F50 (2026-09-10) — the F49 build round: +3.0% bit-identical over shipped; one survivor promoted-ready, one closed null.

All three F49 wrapper/kernel items built and benched, one harness, checksums
identical to shipped on every arm.

* **bf16-I/O gemmseg (VQ_GEMMSEG_BF16IO=1): +1.3-1.8%.** TIO template; both
  boundary casts deleted; BIT-IDENTICAL by construction (stage-in uses the
  astype's own rounding, store preserves the float->half->bf16 double-round).
  182 kernel tests green both flag states. Default off pending the next
  runtime rev decision.
* **Vectorized tile build: +0.5-0.8%,** bit-identical tmeta (200-trial
  equivalence pin). Default ON (it is simply better code).
* **xt ld-padding: NULL — measured tie inside noise** across interleaved
  reps, bit-exact. The proposal priced this outcome; the compiler/hardware
  already handles the 128 B stride. Flag kept (default off), hunt closed.

**Cumulative vs the SHIPPED arc6 bundle, one session, interleaved:** 2151.5 /
2157.3 -> 2215.5 tok/s = **+3.0%**, moving 35B-3.4 from 82% to **84.5% of
affine-8bit prefill**. Remaining measured headroom to ~95%: the rest of the
F49 wrapper (dispatch is dead, scatter small — mostly the broadcast/epilogue
residue) and the ~8% kernel-body gap to affine's qmm. Decode nocast (+3.6%,
F46) still awaits quality gates.

## F51 (2026-09-10) — single vs double round: dead tie on speed; single-round + bf16-I/O locked as the v2 runtime default.

Three interleaved pairs on the idle M3: single 2119.7-2127.2, double
2124.4-2136.1 tok/s — fully overlapping. The epilogue convert is buried in a
bandwidth-bound store, exactly as predicted. Quality (F50 gates): single
measured −0.05% NLL (referee) and +0.6% of existing KL damage (Flash teacher,
single 2048-token sample) — consistent with zero-mean noise from 48 layers of
moved last bits; not provable either way on this corpus. Decision (Noah):
single-round for elegance, PPL retested at the v2 model release as the
control. bf16-I/O now DEFAULT ON in the repo runtime; VQ_GEMMSEG_BF16IO=0 is
the kill switch. The shipped arc6 fleet is unchanged until v2.

Gate-tooling debt found en route: the flashnext teacher cache predates
kl_damage.py's current schema (meta keys + a 2049-vs-2048 token off-by-one;
causality makes trimming safe). A shim lives at scratchpad/flash_teacher_shim;
regenerating fleet teacher caches in the current format, with MORE than one
2048-token sample, is a prerequisite for any future numerics gate that needs
error bars.

## F52 (2026-09-10 evening) — the runner crash SOLVED: per-request Metal buffer accrual, reached only by ~60k-step generations. F49-addendum's "byte-clean" probe was 30x too short.

A third `[metal::malloc] Resource limit (499000) exceeded` crash, this time
with the request in the log: an extraction call went out with
max_output_tokens=65536 (the tools path sent num_ctx — the 09-05 clamp
removal left it unbounded) and a non-verbatim rambler ran ~60k decode steps
before the runner died. 499000 is a COUNT limit: at ~8 small Metal
allocations per decode step, ~60k steps ≈ 500k buffers. This reconciles
everything: the F49-addendum probe (2000 steps, bytes only) was clean because
it was 30x too short and watching the wrong axis; both overnight VQ-runner
crashes sat under raised/unbounded output budgets; stock models never
crashed because nothing ever let them run that long.

Fixes: the tools path is now bounded at 32k (SCOUT_TOOLS_OUTPUT_CAP;
history-preserving comment carries both incidents). REMAINING UPSTREAM
QUESTION for exo/mlx: whether per-step buffers within one request should
accrue at all — a >50k-step single request is the reproduction recipe if
anyone wants the real fix. Not artifact-implicated; serving-layer +
unbounded-client interaction.

Extraction CLOSED: bank A's final 2 transcripts yielded 4 proposals through
the crash (exo auto-recovered, client retried). Round total: 15 submitted +
13 salvaged = 28.

## F53 (2026-09-10) — the v2 runtime is complete and measured: prefill +3.0%, decode +3.3-3.8%, all in-repo behind kill switches.

Decode bf16-I/O confirmed in the runtime proper: 52.1/52.2 -> 53.8/54.1
tok/s interleaved (VQ_DECODE_BF16IO, default on) — F46's +3.6% reproduced.
The v2 stack as committed: routing memo + vectorized tile build + gemmseg
bf16-I/O (single-round, Noah's call, F51) + decode bf16-I/O. Release
control: PPL retested at the v2 model release (the arc6 fleet is untouched
until then).

**v2 candidates surfaced by bank A's salvage** (extraction round closed at
15 submitted + 13 salvaged = 28):
* `_SRC_FUSED_PACKED_D8_SIMD_SS` — an already-written, already-measured
  simd_sum arm recovering ~16% of d8 simd dispatch, parked behind a policy
  gate whose unlock contract its own comment states (1-ULP + ppl/KL
  re-referee + end-to-end A/B). Found by a worker READING the source.
* Paired MTP-acceptance as a fleet no-harm gate: same prompts, cast vs
  nocast arm, |Δacceptance| within published within-rung sd — zero new
  instrumentation, runs on the 11 sidecar repos.
* Per-geometry nocast flags (d8 float4-staged is the riskiest class — it
  skips fp16 rounding of x entirely under nocast) — the refinement of
  today's global flag if the gate flags any rung.

## F54 (2026-09-10) — KERNEL-BODY arm 1 lands: output-block pairing, +5.1-6.6% prefill, bit-exact. The v2 stack reaches ~87-88% of affine.

Swarm7's unrefuted staging-amortization proposal, built: each threadgroup now
owns two 32-column output blocks — the gathered xt slab staged ONCE per group
step (was once per out-tile: 24x affine's staging traffic at OUT=768), wtT
decoded serially per block (threadgroup bytes unchanged), grid.x halved.
Single code path: at OT2=0 the ob loop compiles to the prior kernel exactly.

Interleaved on 35B-3.4: off 2146-2159, ON 2254-2302 tok/s = **+5.1-6.6%**,
checksums identical. Largest single kernel win since CB_DEV, and the first
ever sourced from reading the competitor (affine stages X once per BM tile —
this closes that structural difference).

**v2 stack cumulative vs shipped arc6 (2151-2157): ~2277-2302 = +6-7%
prefill**, decode +3.3-3.8% — putting 35B-3.4 at **~87-88% of affine-8bit**
from 82% shipped. Remaining campaign arms: 1.5 (barrier pipelining), 2
(direct epilogue store), 3 (staging shape) — plus the SIMD_SS unlock for d8.
Implementation note for arm 2: the per-block epilogue now runs twice; a
direct store deletes ybuf AND both epilogue barriers, so the two arms
compound.

## F55 (2026-09-10 evening) — arm 2 (direct epilogue store): bit-exact, NULL on speed. And a pricing-methodology lesson with numbers.

Built with steel's lane mapping (mlx mma.h fm/fn fragment coords — correct on
first compile, checksums identical). Interleaved: off 2261/2280, ON 2261/2279
— dead tie, despite the deletion arm pricing the scalar copy at ~+4.9%.

**The lesson, quantified: a deletion ceiling bounds REMOVAL, not
REPLACEMENT.** The direct store's predicated per-lane scatter costs about
what the staged+coalesced ybuf copy did; what the deletion arm measured was
the cost of writing y AT ALL, not the ybuf detour. (Also: the first pricing
attempt anchored one accumulator and DCE deleted 7/8 of the MAC chains —
+14.6% of pure artifact, rule-4's third catch of the day.)

Code kept default-off (VQ_GEMMSEG_DSTORE): bit-exact, -4 KB threadgroup,
possibly useful for future large-K threadgroup-codebook geometries.

**Campaign scoreboard: arm 1 +5.1-6.6% (landed, default on), arm 2 null
(closed), arm 1.5 (barrier pipelining) next — now MORE interesting since
OT2 serialized the per-group sequence — then arm 3 and the SIMD_SS unlock.**

## F56 (2026-09-10 night) — campaign arms 1.5 and 3 settled; the v2 stack hits +11.9% over shipped, ~91.5% of affine.

**Arm 1.5 (prefetch pipeline): NEGATIVE, −1.8-2%** (off 2308-2322, ON
2271-2277, bit-exact). The salvaged design's own risk clause named it:
register pressure, or the compiler already pipelines those loads and the
explicit version broke its schedule. Flag kept as record, default off.

**Arm 3 (staging shape): +3.7-3.9%, bit-exact, default ON.** Predicate
hoisted to one outer branch, staging vectorized at D_BAKE width. The
refutation was RIGHT about the wrong thing: it killed the traffic claim
(bytes indeed unchanged) while the true mechanism — instruction count and
hoisted predication — carried the win. A verdict of "flawed" attaches to a
CLAIM, not an arm; the campaign ran it anyway because the mechanism was
separable, and it paid.

**Cumulative, one session, interleaved: SHIPPED 2142.5/2142.8 → V2
2398.3/2395.3 = +11.9% prefill** (plus +3.3-3.8% decode). 35B-3.4 stands at
**~91.5% of affine-8bit** from 82% shipped. Campaign ledger: arm 1 +5-6.6%,
arm 3 +3.7-3.9%, arms 1.5 and 2 measured null/negative and closed. Remaining
headroom to affine: ~8.5 points — candidates: the SIMD_SS d8 unlock (its win
is d8-specific), a replacement-priced epilogue idea if one appears, and the
decode remainder (launch census still unrun).

## F57 (2026-09-10 late) — clean same-session parity map: 90.5% prefill, 85% decode; two of my predictions corrected by the instruments.

Head-to-head, both 35B builds local, one session, interleaved.

**Prefill ladder (MoE-block -> x*0 arm on BOTH builds):**
| | VQ-3.4 (v2 runtime) | affine-8bit |
|---|---|---|
| baseline | 3.77 s (2386 tok/s) | 3.41 s (2637) |
| expert block deleted | 2.16 s | 2.15 s |
| **expert block cost** | **1.61 s** | **1.26 s** |

Trunks are IDENTICAL (2.16 vs 2.15). Corrections this forces:
* "Expert path already at parity" (my cross-day arithmetic) — WRONG. The
  expert block is **1.28x** affine's; the entire end-to-end gap (0.36 s) is
  expert-side. The kernel-body campaign is at ~90.5%, not done.
* The 6-bit-trunk theory (F57-eve audit: our attention IS 6-bit vs their
  8-bit) — measured IRRELEVANT: equal trunks. F49's 0.31 s "trunk gap" was
  cross-day drift + arm-scope (F41's stub left shared_expert in the
  remainder; the MoE-block stub does not). No quant-config change needed.

**Decode head-to-head: affine WINS, 63.2-65.0 vs 54.0-54.5 tok/s (~1.18x).**
My bytes-argument prediction inverted: at batch-1 the VQ decode expert
kernels are dependent-gather LATENCY chains (F48's Flash mechanism, now
confirmed to bind on d4-K2048 too) and latency beats bandwidth. The dense
27B pair still favors VQ in serving (different decode path) — MoE decode is
the front line.

**Standing after the campaign:** prefill 82% -> 90.5% (+11.9% banked,
bit-exact), decode ~85% of affine on this pair. BEATING parity remains
physically available (fewer bytes moved in both phases) but the fronts are
now: prefill — another ~0.35 s out of the expert block (SIMD_SS unlock,
next affine-grounded round); decode — break the dependent-gather chain
(fixed-code arm first to confirm mechanism, then prefetch-across-j /
gemmseg-style decode tiles). Both busted predictions are recorded here
deliberately: the record should show what we believed and when the
instruments corrected us.

## F58 (2026-09-11) — decode gather chain SOLVED at the d4 tier: the cost was redundant code-word extraction, not the gather. Bit-walker fetch ships default, 1.12x decode, bit-exact.

The fixed-code deletion arm (F57's named next step) ran first: pinning every
packed-kernel code lookup to entry 0 — which lets the compiler delete the
index load AND the extraction ALU AND turns the codebook gather into a
broadcast — is **1.30x decode** (54.3 -> 70.5 tok/s, 35B-3.4, d4-K2048, M3,
200-step manual loop, 2 interleaved passes, spreads <1%). The chain is real.
Then three discriminating arms split the confound, all same harness:

    arm                          index loads  extraction  random gather   result
    1  full pin (0u & code)        deleted      deleted     deleted       1.30x
    pipe (q-block-ahead fetch)     kept         kept        kept          NULL
    2  gather pinned (& runtime-0) kept         kept        deleted       NULL
    3  synthetic codes (hash)      deleted      deleted     KEPT          1.30x

Mode 3 recovers the ENTIRE deletion win with the random threadgroup gather
retained, and mode 2 recovers none of it with the chain retained. So: NOT
bank conflicts, NOT load latency (the pipeline overlap would have hidden
that), but the **work of VQ_CODE itself** — the macro re-reads its 32-bit
device word for every code it extracts (~3 codes/word at BITS=11) and redoes
the shift/or/mask each time, per output row. The d2 kernel's comment already
knew this disease ("the generic VQ_CODE path re-reads a word per code");
this is the first measurement of what it costs at decode: ~23% of the step.

THE FIX — `_SRC_FUSED_PACKED_D4_WALK`, default ON (`VQ_D4_WALK=0` reverts):
walk the code row once, each word loaded exactly once into a 64-bit
bit-buffer, codes shifted out sequentially. The pack layout (32-code blocks
x BITS words, LSB-first, padded tail) makes the walk word-aligned at every
block boundary. Same code values, same dot operands, same ((d0+d1)+d2)+d3
summation shape -> **bit-identical** (60-step greedy logits-checksum
transcript matches the VQ_CODE kernel exactly; an earlier draft that summed
per-code instead of per-quad produced identical TOKENS but drifted the
checksum — association matters, transcripts catch it).

**Measured: 55.0 -> 61.7 tok/s = 1.12x decode, both passes.** A 32-bit
two-word walker variant TIED the ulong version (61.6 vs 61.9), so 64-bit
emulation is not a cost. The ~8 tok/s between walker (61.7) and bound (69.9)
is mostly the mandatory code-word reads the deletion arms also removed —
deletion ceilings bound REMOVAL, not replacement — so 1.12x is close to the
honest ceiling for this front at d4.

OPEN: the same disease exists in every other VQ_CODE consumer — the packed
d8 kernels (Flash/397B decode geometries; d8-K16384 rows are BITS=14,
~2.3 codes/word) and the packed d2 path. Porting the walker there is the
next decode arm; needs the cluster rungs, so it waits for the M4's return.
Prefill (gemmseg) has its own phase-1 fetch and was not touched; the
gemmseg-style decode-tiles idea from F57 is superseded at d4 by this
simpler, measured mechanism. Probe scaffolding (fixed-code modes, the null
pipeline kernel) was measured, recorded here, and DELETED from the runtime.

## F59 (2026-09-11) — the d8 decode fetch front is CLOSED: the walker disease is d4-tier-specific, and d8's remaining ~7% is the device codebook gather, not extraction.

Porting F58's walker idea to the d8 tier, all on Flash-2.1 (mixed d8, M3,
same 200-step harness as F58, interleaved passes):

    arm                                        result
    RB re-measured vs today's DEVX_SS default  0.90x  (09-02 negative REPRODUCES)
    RB + DEVX + SS composition (funnel fetch
      on the fair device-x/simd_sum body)      NULL   (18.95 vs 18.95, BIT-EXACT
                                                       vs the shipping default)
    fixed-code deletion bound (scratch-bundle
      patch, chain+extraction+gather deleted)  1.074x (18.9 -> 20.3, both passes)

The composition was pure text: RB shares staging/dot/reduction text verbatim
with the base SIMD kernel, so the existing devx and ss rewrites apply on top
(one dispatch gotcha: the SPG_C template constant was gated on an exact
`endswith("_d8_simd_rb")` — a composed name silently compiled without its
compile-time SPG and crashed; worth remembering for any future twin).

ATTRIBUTION BY ELIMINATION. The deletion bound removes three things; the
funnel graft removes one (redundant extraction) and captured NONE of the
bound, so the ~7% lives in what deletion alone removed: the RANDOM DEVICE
codebook gather (d8-K16384's 256 KB cb cannot be threadgroup; pinning c=0
turns the gather into a cached broadcast). This independently re-lands the
old profiling arc's number — shrinking the hot set to 4 KiB bought 8%, which
was recorded then as refuting codebook residency as a lever. Two harnesses,
two years of kernels, same 7-8%: the d8 VQ decode path is at its floor short
of changing where the codebook lives, which K16384 forbids.

WHY d4 AND d8 DIVERGE. The d4 thread-per-row kernel extracts every one of
its ~512 codes alone, re-reading each device word ~3x -> extraction was 23%
of the step and the walker recovered half (F58). The d8 SIMD kernel gives
each lane 8 codes and reads x from device (arc 5); its extraction is already
amortized and its codebook is device-resident. Same macro, different
economics. The walker is a d4-TIER fix, correctly shipped there and only
there; RB stays OFF and the RB+DEVX+SS composition is recorded here and NOT
kept in the runtime.

CONSEQUENCE FOR THE DECODE PROGRAM. Flash-class decode improvement now has
to come from the non-VQ trunk (F48: GDN 29%, attention 25%) or from serving
(MTP), not from VQ kernels. The 397B d8 rungs inherit this conclusion —
same geometry, same kernels.

## F60 (2026-09-11) — the prefill "SIMD_SS unlock" was stale, and the phase-1 fetch front closes the same way d8 decode did: the residual is the CB_DEV gather, which is the occupancy win's price.

Two findings, one sitting, both on 35B-3.4 (art_xtpad, M3, memo_check 9k
prefill, interleaved passes):

**1. The named front was already shipped.** F53/F57's "SIMD_SS d8 unlock"
candidate — surfaced by a salvage worker reading the kernel comment — was
composed into the shipping DEVX_SS default on 2026-09-02, gate already
relaxed to 1-ULP by recorded decision. There is nothing to unlock; the
worker read a pre-composition comment. ([[vq-swarm-yield]]: anchors good,
conclusions bad — now with a concrete mechanism: a stale comment is a
perfect anchor for a wrong conclusion.)

**2. The real prefill front (0.36 s expert block) got the F58 treatment.**
Deletion bound: pinning gemmseg phase-1's code fetch (scratch-bundle
_PACK_FETCH patch) is **1.061x** prefill (2350 -> 2493 tok/s, both passes)
— 0.22 s of the 0.36 s gap. A per-segment bit-walker (seeded at each lane's
SPG/4-aligned segment, one load per word, GSWALK template arm) came back
**bit-exact (4096-tok logits checksum identical) and NULL on speed** (2358
vs 2354, 2381 vs 2392). Same elimination as F59: the bound is not the word
re-reads, it is the RANDOM DEVICE CODEBOOK GATHER — cb[c] is device-resident
here because CB_DEV at 16 KB is a 1.46-1.76x occupancy win (F32/F34), and
the deletion arm's c=0 turns that gather into a cached broadcast.

**The CB_DEV gather tax is now measured three independent ways** — ~8%
(2026-09-02 residency probe, d8), ~7% (F59, Flash decode), ~6% (here, d4
gemmseg prefill). It is the structural price of the occupancy win and no
fetch-side rewrite touches it. The walker (F58) remains correct where it
shipped: the d4 thread-per-row DECODE kernel keeps its codebook in
THREADGROUP memory, so there the fetch was the cost and the fix landed.

CONSEQUENCE. Prefill standing stays 90.5% with the honest decomposition:
0.22 s of the 0.36 s gap is the CB_DEV gather (structural), ~0.14 s is
unattributed phase-2/3/epilogue residue the campaign already mined with
five arms. Beating the gather means beating the occupancy trade, not the
fetch: the only named idea left is a smaller-footprint codebook encoding
(fewer bytes per entry), which changes the artifact format and is out of
runtime scope. The prefill kernel front is CLOSED at this ceiling for the
current format. Probe scaffolding (bundle pin, GSWALK arm) deleted after
recording.

**F60 addendum — post-walker decode parity re-measured (same session,
interleaved, F57 protocol):** VQ-3.4 59.6/60.1 vs affine-8bit 63.9/64.0
tok/s = **93.6% of affine decode** (was 85% at F57; affine's lead cut from
1.18x to 1.07x). The residual 1.07x is consistent with the CB_DEV gather
tax on the modules the walker does not serve plus the irreducible code
reads. Trunks were not re-decomposed — F57's equal-trunk result stands.

## F61 (2026-09-11 evening) — the int8-codebook format lever is LIVE: quality marginal-but-refereeable, and the occupancy objection is measured DEAD.

Two probes, both cheap, both on 35B-3.4 (M3):

**Quality (in-memory, no repacking):** every codebook quantized to int8 in
the loaded model, logits compared to the fp16-codebook baseline over 2048
positions. Per-dim symmetric scales win clearly over per-codebook:
rel-F codebook error ~0.52%, **top-1 agreement 99.95%, KL mean 0.012**
(max 3.5 — real tail spikes), teacher-forced nll +0.002 nats absolute.
Verdict: NOT free, plausibly acceptable — the v2-release ppl/KL referee is
mandatory, per-rung, exactly as Noah's release policy already requires.

**Occupancy (the objection that would have killed it):** the CB_DEV gather
tax (F60) is the price of evicting the 16 KB codebook; a resident 8 KB int8
table must not re-pay what eviction bought. Dead-threadgroup-pad probe
(OCCPAD bytes allocated + guard-touched in gemmseg, cb still device):
**8 KB costs 0.0%** (2413 vs 2414 tok/s), 16 KB costs 0.6%. Hardware
occupancy at these sizes is a non-issue — which also reinterprets the old
CB_DEV@16KB=1.76x win as a Python-side FITS-BUDGET routing cliff (over-
budget tiles fell off the gemmseg path entirely), not smooth hardware
occupancy. The int8 lever's upside is the ~6% gather bound (F60) minus the
in-kernel dequant cost, for an artifact-format rev gated on the referee.
NEXT (v2-release scale, Noah's call): int8-cb packer + kernel arm + full
per-rung referee.

## F62 (2026-09-11 evening) — launch census finally run: a decode step is 24% host graph-build and ~3 us/launch — and production serving ALREADY HIDES most of it.

Census on the 35B decode step (manual loop, M3): host graph construction
(forward built, not evaluated) is **4.0 of 16.5 ms/step = 24.4%**; injected
dependent elementwise ops price a launch at **~3.0 us** (slope over k=0..400,
1.19 ms per 400 ops). At ~1000+ ops/step that is the launch-bound decode
signature F47's +7% super-additivity hinted at.

BUT: the manual ladder loop SERIALIZES build and GPU. Production
stream_generate (async_eval pipelining) measured **69.3 tok/s vs the
ladder's 61.7** on the same model — mlx_lm already overlaps most of the
host-build behind the GPU. Consequences: (1) ladder tok/s UNDERSTATES
production absolute throughput ~12% (parity RATIOS stand — both builds were
measured on the same loop); (2) the host-build lever is largely already
pulled upstream; what remains for an mlx/mlx-lm PR is op-count reduction
(fewer, fatter kernels per layer) and the GPU-side gap share, bounded by
F47's ~7%. Capture note for next time: MTL_CAPTURE_ENABLED must be set
BEFORE process start, not after import.

## F63 (2026-09-11 night) — CORRECTION to F60/F61: the prefill fetch bound is not the gather's LOCATION, it is the mandatory fetch work. The int8-codebook lever is DEAD for speed.

The decisive arm existed all along: `VQ_GEMMSEG_CBDEV=0` forces the 16 KB
d4-K2048 codebook back into THREADGROUP memory (legal at 29 KB total), i.e.
full residency with zero format change. Measured (memo_check, interleaved,
checksums identical = bit-exact): **2400 vs 2412 tok/s — +0.5%, noise.**
Residency recovers nothing.

Elimination now runs to the end. The F60 deletion bound (1.061x) removed
three things; each has now been isolated:
  - redundant word re-reads — walker arm, NULL (F60);
  - gather location / randomness — residency arm, NULL (here);
  - the MANDATORY index loads + extraction ALU + dependent chain — the
    remainder, and therefore the whole 6%.
Only deleting the codes deletes that work. **Deletion ceilings bound
REMOVAL, not replacement** — third bite of that rule this arc (F55 pricing,
F59 d8, now F60's attribution), and this one propagates: F61's int8-codebook
speed case was "recover the gather tax by residency"; residency is now
measured worthless, and the size savings are negligible (codebooks are
~0.01-0.2% of artifact bytes). The int8 lever is CLOSED — do not spend the
quality budget. (F61's occupancy measurement stands and remains useful: 8-16
KB of threadgroup residency is nearly free in gemmseg2, unlike the fused
kernels — the ≥16 KB CB_DEV rule was calibrated on the wrong kernel's
sensitivity, it just happens not to matter either way.)

STANDING. Prefill 90.5% of affine: ~6% is irreducible fetch work inherent
to reading packed VQ codes, the rest is campaign-mined residue. Decode
93.6%. The VQ kernel program is closed at the honest floor for ANY codebook
format — the remaining absolute wins are the non-VQ trunk (upstream mlx
material, F47/F48 decompositions) and MTP serving.

## F64 (2026-09-11 night) — batch-decode curve, VQ vs affine, B=1..8: the ratio improves to ~93% by B=2 and PLATEAUS. My amortization prediction graded: half right.

Same manual loop, both builds interleaved at each B (35B pair, M3, 100
steps, best of 2 warm reps):

    B   VQ agg     affine agg   ratio
    1    61.2        68.6       89.2%
    2   111.7       119.9       93.2%
    4   182.2       196.3       92.8%
    8   297.2       324.4       91.6%

Prediction on record was "ratio climbs with B toward prefill-like levels as
the fetch tax amortizes across rows." Graded: the B=1 -> B=2 jump is real
(+4 points), then FLAT — no climb toward 100%. The mechanism I missed is
MoE routing: batched rows route to DIFFERENT experts, so weight tiles are
shared only where rows happen to overlap — decode batching amortizes the
per-weight fetch far less than prefill's 32-row tiles do. (Prefill re-uses
every tile across all rows by construction; decode B=8 mostly adds expert
work proportionally.) Both builds scale sublinearly for the same routing
reason (affine/row drops 68.6 -> 40.5 too); VQ neither catches up nor falls
behind.

SERVING TAKEAWAY. For multi-agent workloads VQ holds a stable ~92-93% of
affine at practical batch sizes while resident at 2.4x smaller — the
multi-agent case rests on residency (model + KV headroom), not on a
throughput crossover, and no crossover should be claimed. Note the absolute
aggregate: 297 tok/s at B=8 from a 15 GiB artifact.

## F65 (2026-09-11 late) — per-expert codebooks measured and CLOSED: expert subvector distributions are homogeneous; the shared codebook is the right architecture by ~measurement, not just by default.

PoC on real weights (layer-10 gate_proj of the 35B, E=256, dequantized
affine-8bit as fit target; identical treatment both schemes: G=64 max-abs
scales, d4 subvectors, same k-means, full-assignment reconstruction):

    shared  d4-K2048   rel-F 0.1862   11 bits/subvec
    per-exp d4-K256    rel-F 0.3087   8 bits   (1.66x worse)
    per-exp d4-K1024   rel-F 0.2179   10 bits  (1.17x worse)
    per-exp d4-K2048   rel-F 0.1822   11 bits  (0.978x -- 2.2% better)

The specialization ceiling is ~2% error at EQUAL bits, for 256x the
codebook storage (~4 MB/module, ~0.5 GB model-wide); at fewer bits
per-expert always loses. The MoE intuition ("only the called-on vocabulary")
does not transfer: experts differ in WHICH weights they hold, not in the
statistics of their subvectors, and codebook fitting sees only the
statistics. The byte-aligned-K256 speed synergy dies with it (that lever
was already dead independently via F63). Caveat inherited from the
fit-proxy rule: this is weight-space recon, not KL — but at 1.17-1.66x
error deficits the direction is not in doubt. Related live idea kept
separate: per-PROJECTION-TYPE sensitivity-scaled K (gate/up/down), which
shares nothing with this negative and still awaits its sensitivity map.

## F66 (2026-09-11 late) — per-projection sensitivity map: compressibility is IDENTICAL across gate/up/down; sensitivity orders gate > down > up (1.55x spread). The allocation axis is real but THIN (~0.08 bpw).

Two probes, layer-10 module trio + whole-model noise injection (35B):

Compressibility (shared-codebook k-means ladder, identical treatment):
    proj        K256      K1024     K2048
    gate_proj   0.30989   0.22078   0.18634
    up_proj     0.31034   0.22058   0.18630
    down_proj   0.30935   0.22053   0.18605
Statistically identical to the third decimal — the F65 homogeneity result
extends across projection TYPES: nothing in the expert stack is easier to
quantize than anything else.

Damage per unit error (0.5% relative codebook noise, one type at a time,
KL vs clean over 2048 positions): gate 0.0144 > down 0.0124 > up 0.0093 —
gate most fragile (against the down-proj classic prior), 1.55x spread.

ALLOCATION CONSEQUENCE. With equal compressibility, allocation = starve
the insensitive: demote up_proj one K notch (K1024) for ~0.08 bpw at 1.18x
error on the least-sensitive third. Real, small, and only worth bundling
into a v2 fit campaign that is happening anyway — not worth its own. The
strong version of the axis (differential compressibility) is measured
absent. Caveats: single layer for compressibility (homogeneity so far
suggests it generalizes), noise-KL is a perturbation proxy (directionally
robust at 1.55x, per the F65 caveat discipline).

## F67 (2026-09-11 night) — the MSE-VQ cap is real and the escape is measured: activation-aware code re-selection wins +10.7% output-space error HELD-OUT. Tables-only is nearly worthless; the prize is the codes.

PoC on layer-10 gate_proj (E=256), real activations captured from the
running 35B (4096 tokens of diverse technical text, split train/test;
optimize on train Gram, score on test Gram):

    arm                                   train G     test G (HELD-OUT)
    k-means (Lloyd, weight-MSE)           baseline    baseline
    + table tuning only (Adam, frozen     +0.51%      --
      codes, output-space loss)
    + G-aware code re-selection           +10.17%     **+10.70%**
      (block-diagonal Gram metric)
      then table retune

    (weight rel-F WORSENS 0.1875 -> 0.1938 while output error drops --
     proxy inversion in one row. An earlier repetitive-prompt run showed
     +32%: subspace-inflated, discarded; diverse-text held-out is the
     number.)

Zero train/test gap (10.2 vs 10.7): the method learns the model's
activation geometry, not a corpus. The decomposition matters: with a
diverse Gram, tuning TABLES with frozen codes recovers ~nothing (Lloyd's
tables are nearly fine) — the cap is in the ASSIGNMENTS, exactly the
variable k-means assigns under the wrong (unweighted) metric. Cost of the
whole pipeline on this module: ~20 s.

CONSEQUENCES. (1) The KL-TUNED-CODEBOOKS-PLAN's phase priorities INVERT:
code re-selection is phase 1, table tuning a garnish. (2) This spends the
"data-free" claim: v2 becomes lightly-calibrated (AQLM-class); ship the
data-free fit as base recipe + calibrated code-refinement as the
documented upgrade, calibration set named on the card. (3) The 397B scar
applies (activation-fitted quant FAILED there, DWQ/GPTQ arc): different
mechanism (codes re-selected within fixed geometry, weights untouched),
but graduation is end-to-end ppl/KL per rung ONLY — reconstruction
proxies are the thing that inverted last time. Next: whole-model
re-selection on 35B-3.4 + real KL referee; then the per-rung sweep on the
v2 train.

## F68 (2026-09-11 night) — SELF-CALIBRATION works: the model's own generations recover 88% of the corpus-calibrated win (+9.4% held-out vs +10.7%) from 1,395 self-generated tokens. The data-free claim is RESTORED.

Same pipeline as F67 with one swap: the training Gram comes from text the
35B free-generated itself (temp 1.0, four one-word seeds, ~1.4k tokens,
re-forwarded for capture). Scored on F67's REAL-TEXT held-out Gram:

    calibration source        held-out gain vs k-means
    real diverse text (F67)   +10.70%
    SELF-GENERATED            **+9.41%**   (88% of the corpus win)
    (tables-only, either)     ~+0.5%

Noah's insight verbatim: "the model will find its own activations." The
activation second-moment geometry is a property of the MODEL, and the model
samples its own distribution by construction — no corpus curation, no
domain-skew exposure to an external dataset, no data shipped or named. The
v2 recipe is therefore fully self-contained: pack -> self-generate ->
capture -> re-select codes -> referee. The corpus question (F67
consequence 2) largely dissolves; a mixed external corpus remains the
+1.3%-better option and the cross-domain-skew probe still applies to BOTH
(self-generation at temp 1.0 has its own distribution bias toward the
model's high-probability modes — the per-domain referee stays mandatory).
Cost: generation dominates; the re-selection itself is still ~20 s/module.

## F69 (2026-09-12 early) — self-calibration token-scaling: saturates at +9.4% by 8k tokens; the 1.3-point gap to corpus is DISTRIBUTIONAL, not statistical.

Cumulative self-generation ladder (20 diverse seeds incl. code tokens,
temp 1.0), G-aware re-selection per rung, all scored on the F67 real-text
held-out Gram:

    N tokens    held-out gain
      2,048       +9.03%
      8,192       +9.35%
     32,768       +9.40%
    131,072       +9.41%
    corpus ref   +10.70%   (F67)

Textbook second-moment convergence: 16x more tokens past 8k buys +0.06
points. The residual 1.3 points vs corpus calibration cannot be bought
with samples — it is the temp-1.0 generation distribution under-covering
part of real-text activation geometry (mode bias). Levers if that last
point ever matters: seed diversity/temperature schedules, or just use a
corpus (F67). RECIPE SETTLED for the v2 campaign: ~8k self-generated
tokens (~2 min GPU on the 35B), Gram, re-select, referee — data-free,
held-out-validated, saturation-verified. F58-F69 close the arc.

## F70 (2026-09-12 early) — calibration showdown on TWO test domains: the corpus's edge was HOME-FIELD; on neutral ground plain self-generation WINS. Steered seeds lose to plain sampling.

Five calibration arms, each scored on T1 (findings-log held-out half — same
DOCUMENT the corpus arms train on, disjoint slices) and T2 (code domain:
mlx_lm source, disjoint from everything):

    arm                     T1-prose   T2-code
    corpus 2k               +9.68%     +6.48%
    corpus 8k               +9.91%     +6.65%
    corpus 32k              +9.95%     +6.61%
    self-gen plain 8k       +8.85%     **+7.01%**
    self-gen steered 8k     +8.39%     +6.77%

Verdicts: (1) F67/F69's "corpus beats self-gen by 1.3" was largely
home-field — the corpus and test Grams shared a document; on the neutral
code domain self-generation WINS outright. (2) Corpus calibration
saturates by 2k tokens — second-moment convergence is universal here.
(3) Prompt-engineered seeds (code/table/multilingual starters, mixed
temps) are WORSE than plain temp-1.0 sampling on BOTH domains — the
model's natural distribution mirrors its own activation geometry better
than steering does. RECIPE FINAL: plain self-generation, ~8k tokens,
temp 1.0. Data-free is not a compromise; on neutral domains it is the
better calibration. (Everything above remains layer-10 single-module
output-space; the whole-model KL referee is the running overnight job.)

## F71 (2026-09-12 morning) — WHOLE-MODEL VALIDATION on the 35B: self-calibrated code re-selection improves end-to-end KL, tails most, benchmarks concur. The 397B activation-fitting scar does NOT repeat at model scale.

The full recipe ran overnight on 35B-3.4: 8k self-generated tokens ->
per-module activation Grams (all 120) -> G-aware code re-selection ->
artifact assembled on disk (art_reselect, scratch) -> refereed against
cached affine-8bit teacher top-256 logits on held-out text:

    metric                     original     re-selected
    KL vs teacher (mean)       0.101311     0.099082    (-2.2%)
    KL worst position          4.449        3.755       (-15.6%)
    top-1 agreement            83.69%       84.20%      (+0.51 pts)

Task benchmarks, card protocol (1000 items, 0-shot, lm-eval 0.4.12
rebuilt-pinned, same harness family as the published numbers):

    task         original    re-selected
    HellaSwag    0.741       0.735
    PIQA         0.828       0.835
    WinoGrande   0.736       0.747

Net +1.2 points across three tasks; per-task swings are within ~1.4-pt
sampling noise, but ensemble + KL + tail direction all agree: a SMALL REAL
IMPROVEMENT, zero cost (same bits, same tables, same runtime, codes only).
The worst-position KL improving MOST (-15.6%) is the shape you want —
quantization damage lives in the tails.

Caveats standing: teacher is the affine-8bit (also the fit target); the
campaign-grade referee is the bf16 teacher via the streaming path. Referee
text is one 4k slice; ppl sweep at v2 release per Noah's policy.

OPS LESSONS (cost a night and a frozen Mac):
* A background chain tied to the Claude session dies with it — overnight
  jobs get nohup+disown and stage checkpoints, ALWAYS.
* Flash-Next-8bit is 178 GB (a ~160B model) — residency is impossible on
  ANY box; the layer-streaming scorer pattern computes teacher logits in
  ~2 min at ~4 GB peak (arch-aware port needed for qwen4_exp: rope module,
  ssm/attention masks, PLE prev_ctx, hyper_connection_mixer, NO final norm).
* Two silent OOM kills + one machine freeze from re-selection beside a
  resident student: stage R now runs STANDALONE (no model loaded) under
  mx.set_memory_limit, per-module checkpointed. A Metal GPU-hang watchdog
  trips if >~64 K16384 argmin steps queue lazily — eval every 8.
* The bench venv died in the quantlab->vqlab merge; rebuilt pinned
  (lm_eval 0.4.12 + mlx-lm 0.31.3) at scratchpad/bench_venv.
* PIPELINE ORDER (Noah's question, settled): sensitivity/geometry mix ->
  k-means fit -> G-aware re-selection -> referee. Re-selection is a cheap
  polish INSIDE a fixed geometry; if Flash shows the gain is
  geometry-lopsided, score mix candidates WITH re-selection applied.

IN FLIGHT: Flash-2.1 generalization (d8-K16384, GDN+PLE arch, 4x scale) —
teacher streamed, 144 Grams banked, baseline KL 0.4335 / top-1 71.46%,
re-selection running checkpointed. Its verdict decides whether F67's
recipe is lineup-wide or d4-specific.

## F72 (2026-09-12) — Flash-2.1 generalization: the recipe transfers to d8-K16384 (KL -1.7% mean, top-1 +0.22), but the MAX-KL rose. Mixed verdict; the win is smaller and less clean than the 35B's d4.

Whole-model re-selection on Flash-Next-VQ-2.1bpw (138 d8-K16384 modules
re-selected; the 6 d2/K256 modules in layers 0-1 kept original — different
storage format, and NOT the geometry under test). Streamed bf16-adjacent
affine-8bit teacher (178 GB model, layer-streamed at ~4 GB peak), held-out
referee slice:

    metric                  original     re-selected
    KL vs teacher (mean)    0.433488     0.425984    (-1.7%)
    KL worst position       6.872        7.469       (+8.7%  WORSE)
    top-1 agreement         71.46%       71.68%      (+0.22 pts)

READ: the mean and top-1 move the right way, so the mechanism DOES transfer
across 4x scale, GDN+PLE arch, and the d8-K16384 geometry — it is not a
d4/35B special. But the improvement is ~half the 35B's mean gain and the
WORST-POSITION KL got WORSE (opposite of the 35B, where tails improved
most). Two candidate causes, unseparated: (1) the teacher here is the same
affine-8bit whose 2.1bpw student is far from it (baseline KL 0.43 vs 35B's
0.10) — re-selection pulls the bulk toward the teacher but can trade a few
tail positions; (2) block-diagonal Gram re-selection at d8 ignores
cross-subvector correlation the d8 kernel's simd layout may couple. Not
chased tonight.

CONSEQUENCE FOR v2: the recipe is lineup-applicable but must be
PER-RUNG-REFEREED on quality, not assumed — the 35B's clean tail win does
not guarantee itself at lower bpw / larger scale. The mean+top-1 gains are
real and free; whether they clear a v2 quality bar is the release referee's
call (ppl sweep, per Noah's policy), especially given the max-KL regression.
Method note: the d2 layers-0-1 need format-matched re-selection (unpacked
uint, not bit-packed) before they can be included; skipped here as
out-of-geometry.

## F73 (2026-09-12) — the F72 "tail regression" DISSOLVES under per-position instrumentation: the max-KL rise is ONE position; the tail as a distribution IMPROVED. Flash behaves exactly like the 35B.

F72's alarming number (worst-position KL +8.7%) was a single order
statistic. Per-position KL vectors for both arms (same student load, same
cached teacher, 4096 positions; scratchpad/tail_probe.py ->
tail_probe_kl.safetensors):

    quantile   original    re-selected
    P50        0.2715      0.2740     (+0.9%)
    P90        1.0171      1.0093     (-0.8%)
    P99        2.6095      2.6202     (+0.4%)
    P99.5      3.5290      3.2541     (-7.8%)
    P99.9      5.0250      4.6026     (-8.4%)
    max        6.8717      7.4692     (+8.7% -- position 2358, ' defects',
                                       5.52 -> 7.47, the ONLY point above P99.9
                                       that regressed this much)

Churn is real but favors improvement at every magnitude: |dKL|>0.5:
158 worse / 176 better; >1.0: 31/48; >2.0: 5/10. Decomposed by
baseline-KL decile, the mechanism is IDENTICAL to the 35B's (F71):

    deciles 0-7 (easy bulk)    sum dKL  +99.7   (small broad tax)
    decile 8                   sum dKL  -23.3
    decile 9 (worst positions) sum dKL -107.4   (the prize)

Re-selection repairs the positions the student was worst at, paying a
small tax on easy ones — the tail-repair shape F71 called "the shape you
want," present at d8-K16384/2.1bpw too. F72's hypotheses (a)
cross-subvector Gram blindness and (b) teacher distance are NOT NEEDED to
explain a tail regression, because there is no tail regression to explain.
The max-KL line in F72 stands as written but its READ was wrong: quantiles,
not the max, are the tail instrument (a max over 4k correlated positions
moves ~this much under any code churn).

CONSEQUENCE: the d8 blocker on scaling the recipe is LIFTED. Remaining
before per-rung referee: (1) hypothesis (a) demoted to an upside question —
does full-Gram ICM buy MORE than block-diagonal? (icm_probe.py running,
banked Grams are full IN x IN so this needs no recapture); (2) d2/K256
format-matched re-selection for layers 0-1; (3) bf16 teacher remains
campaign-grade referee hygiene (affine-8bit is also the fit target) but is
no longer suspected of causing a regression; no bf16 Flash exists on disk
(~320 GB download — Noah's call if wanted).

## F74 (2026-09-12) — full-Gram ICM re-selection is DEAD: it overfits the calibration Gram catastrophically (+56% train, −20% HELD-OUT). Block-diagonal is the right operating point, by measurement.

Hypothesis (a) from F72, run as its upper bound (scratchpad/icm_probe.py):
full-Gram coordinate-descent (ICM, 3 sweeps, converging flips) on the
layer-20 probe modules, scored as output-space error J = mean(e^T G e) on
the 8k-self-gen TRAIN Gram (optimized) and a held-out test Gram captured
from disjoint text (40,960 token-rows). The banked Grams are full IN x IN
— re-selection had only ever used the 8x8 diagonal blocks.

    module (E=512, K=16384)     arm         J_train      J_test (HELD-OUT)
    down_proj (IN=640)          blockdiag   +0.44%       -0.26%
                                ICM full-G  +3.86%       -2.63%
    gate_proj (IN=2560)         blockdiag   +13.26%      +2.18%
                                ICM full-G  +56.21%      -19.80%
    up_proj (IN=2560)           blockdiag   +12.10%      +1.08%
                                ICM full-G  +55.46%      -21.66%
    (gains vs k-means codes; positive = better)

MECHANISM. An 8x8 Gram block is well-estimated from 8k tokens; a full
2560x2560 second moment from the same tokens is rank-starved, and ICM
drives the residual into its noisy/null directions — the textbook
overfit signature (train and test move in OPPOSITE directions, huge
spread). The cross-subvector correlation the d8 kernel couples is not
exploitable at this calibration size; chasing it would need order
100k+ tokens of calibration (and F69 showed block-diag saturates at 8k),
so the data-free recipe stays block-diagonal BY MEASUREMENT.

Also visible in the table: Flash's per-module held-out gains (+1-2% on
gate/up, ~0 on down) are far below the 35B layer-10 module's +10.7%
(F67) — consistent with F72's smaller whole-model mean gain at 2.1bpw.
The d8/2.1bpw win is thinner per module, not tail-broken (F73).

Verdict: F72's two hypotheses are both closed — (a) full-Gram is
measured WORSE, (b) is moot because there is no tail regression (F73).
The shipped recipe (block-diag, 8k self-gen) is the validated operating
point. Remaining Flash work: d2 layers 0-1 format-matched re-selection +
whole-model referee (chained, running), then the per-rung ppl sweep.
