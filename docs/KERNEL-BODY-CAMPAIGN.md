# Kernel-body campaign — closing gemmseg's last ~8% to affine's qmm

*2026-09-10. Target from F49: kernel body 1.37 s vs affine's whole expert
path 1.26 s (9k tokens, 35B). Source: swarm7's affine-grounded proposals
(first round to read quantized.h), their refutations, and F51's null.*

## What is now KNOWN about the difference

* **Bank conflicts are NOT it** (F51): the affine-style xt ld-pad measured a
  tie; Apple's toolchain already handles the 128 B stride. The whole
  "conflict" hypothesis family is closed.
* The remaining structural differences, in order of expected size:

## Arm 1 — staging amortization (OTILE 64)  [UNREFUTED; the big one]

gemmseg's grid is one threadgroup per (row-tile, out-tile): the SAME gathered
x slab is staged into xt once per 32 output columns. Affine stages X once per
BM tile. At OUT=768 that is 24x the gathered-x staging traffic. Proposal:
each simdgroup owns TWO 8-wide out blocks (C0..C7), wtT refilled per
out-block with one extra barrier per group — halves staging today, and the
shape generalizes. NSG stays 4 (F37: not a knob). Phase 2 is 9.6% of prefill
(F39); halving its staging is worth up to ~4-5% alone, plus fewer dispatches.
Refuter: FAILED (endpoint death), so this needs the standard bit-exactness +
one-harness bench discipline, not extra faith.

## Arm 2 — direct epilogue store  [core survives its refutation]

Affine ends its K loop `store_result / store_result_safe`: registers ->
device, one fast/safe branch per tile, NO third threadgroup buffer. gemmseg
round-trips C0..C3 through a 4 KB float ybuf, pays a barrier, then re-reads
scalar-by-scalar under a per-element guard. The refuter KILLED the aliasing
variant on its own arithmetic (proposed buffer 2x too small — silent
corruption) — correctly. The surviving form is affine's own: full interior
tiles take `simdgroup_store(C*, y + offset, OUT)` directly; edge tiles keep
the guarded scalar path. Deletes 4 KB of threadgroup (occupancy per F36's
slope ~ +1.2%/KB freed), one barrier, and a full scalar copy pass.
CAUTION (F44's lesson): the store dtype must preserve the shipping rounding
(float accum -> TIO), and CB_DEV budget arithmetic must follow the -4 KB.

## Arm 1.5 — prefetch-pipelined decode (single barrier per g-step)  [salvaged, unrefuted]

From bank B's recovered affine_diff analysis: gemmseg alternates
decode-barrier-MAC-barrier per group step; affine's steel overlaps its next
tile load with the current MMA. The salvaged proposal:

> Issue code and scale device loads for iteration g+1 into registers before the matmul of iteration g (inside phase 3, before MMA loop). After the matmul barrier completes (proving all simdgroups done reading wtT), perform only the dependent codebook gather and write wtT[g+1]. This overlaps independent device loads (code words, scale halves) with MMA computation. Pipeline: issue code/scale loads for g+1 → matmul(g) executes → barrier (both matmul done and prefetch in-flight) → dependent codebook g

Convergence note: bank B independently designed today's bf16-I/O
implementation (decode pass-through + TIO-templated gemmseg with in-kernel
staging) before it existed, and its "direct simdgroup_store epilogue" is Arm
2 verbatim -- two models agreeing with each other and with the built code
raises confidence in this whole arm family.

## Arm 3 — phase-2 staging shape  [refuted on win-size; keep small]

Per-element predicated 2-byte loads vs affine's BlockLoader with hoisted
bounds + vector width. Refuter is right that BYTES don't change; instruction
count and load latency do. Smallest arm; run only if 1-2 leave a residue.

## Discipline

One harness (F41's), bit-exactness gate per arm (checksum for exact arms,
referee for rounding-touching arms), interleaved reps, budget arithmetic
follows every threadgroup-byte change, and each arm lands behind a flag with
the kill switch documented. The pad flag (F51) stays as the template.
