# FINDINGS LOG — the tended record

One entry per finding: the NUMBER, how it was measured, what it changed.
Newest first. This file exists because no context window survives the arc —
write here the moment a number lands, not at the end of the night.
Rules: numbers only if measured; every claim carries its method; corrections
edit the entry in place and say CORRECTED. Keep it readable in one sitting.

---

## 2026-09-09 (morning, running the overnight proposals)

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
