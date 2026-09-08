# Swarm: the remaining prefill gap (GLM-5.3-Flash on the cluster, 2026-09-07/08)

First swarm run on the M3+M4 ring rather than a single box.
Model: `TheDrainFlorist/GLM-5.3-Flash-VQ-3.6bpw`, 2-node pipeline.
Grounding: the shipped `model.py` plus VQGEMM-BENCH, KERNEL-COVERAGE, D8-BENCH.
Target: the ~15% still separating us from the affine-8bit floor
(1914 tok/s at d4-K8192 vs 2246).

Six proposers -> 4 usable proposals -> 3 refuted (the 4th refuter was killed
30 s in, on a proposal that only asks for a measurement).

## Results

### 1. Paired token tiles (RTILE 32 -> 64) — VIABLE
Phase 1 decodes the [32 out-rows x G=64] weight tile once per (out-tile,
group), and that result is IDENTICAL for every token tile — so codes
(WPR*4 B/row), scales (NGRP*2 B/row) and the CB_DEV codebook loads are all
re-read once per 32-row token tile. Pairing two same-expert tiles halves
exactly that duplication; x bytes and MAC counts are invariant.

Predicted: d4-K8192 (the 1914 tok/s target) 4.7 s -> 4.35-4.50 s (+5-8%),
codes ~3.8 -> ~1.9 GB/dispatch, codebook L2 lookups ~19 -> ~9.5 GB.

Refuter verified the premise against v1 source (3025-3060) and the budget
against the code's own formula: paired CB_DEV d4-K8192/d8-K16384 lands at
16384-20480 B against the 32768 cap, >=12 KB spare.

### 2. OTILE 32 -> 64 on the CB_DEV arm — VIABLE, but demoted by its own refuter
Phase 2 re-gathers `xsrc[src_rows]` per (token-tile, out-tile, group):
72000 rows x 32 out-tiles x 4096 x 2 B = 18.87 GB/dispatch. OTILE=64 halves
it to 9.4 GB.

The refutation is the most valuable artifact of the run. It returned VIABLE
and then dismantled the priority claim:

- **"Largest single traffic term" is a TIE, not a win.** On CB_DEV the
  device codebook reads exactly equal the x re-reads (per out-row per group,
  G/D codes x 2D B = G*2 B == one token-row's x slice). Both ~18.9
  GB/dispatch. Halving one of two equal terms leaves the other dominant —
  and that other is only reachable via RTILE, i.e. proposal 1.
- **The predicted win is internally inconsistent.** A full DRAM miss of 18.9
  GB at 819 GB/s is ~23 ms against a ~33 ms dispatch = ~+35%, not +5-10%.
  And "<1% if L2-served" ignores that L2-served reads still burn ~1.3 TB/s
  of L2 bandwidth. A positive result will not attribute between DRAM-miss
  and L2-BW-bound.
- **Unpriced confound**: 20480 vs 12288 B threadgroup allocation may halve
  threadgroups-per-core. Control offered: OTILE=64 with RTILE=16 (14336 B,
  near today's).
- **Prior favors the null**: consecutive threadgroups share a row-tile and
  the in-flight window covers all out-tiles, so x is probably L2-served
  already — plus our own v1->v2 lesson was "ALU choreography, not the byte
  plan".
- Code detail we had wrong: v2's template list (3372-3377) contains no
  OTILE, so it is a source-level constant, not a template knob.

### 3. NSUB=80 ragged tail — VIABLE
`gemmseg_fits` requires `(IN/D) % 32 == 0`, so Flash-2.1's 46 d8 modules
with in=640 (NSUB=80) fall to the legacy path — the only sub-100%-fused
artifact in the fleet.

The refuter found the gate is over-conservative and that the runtime ALREADY
anticipates this case: `pack()` zero-pads the last 32-code block, WPR is
already `ceil(NSUB/32)*BITS` in the source, inner loops are bounded `n <
NSUB` so pad codes are never read, and `input_dims` (3682-3688) uses NSUB=80
as its LITERAL worked example, deriving IN from `vq_scales.shape[2]` exactly
because WPR->NSUB is lossy for padded tails. The metadata path exists; only
the gate blocks it.

Predicted: Flash-2.1 fused arm 11.59 s -> 10.5-11.1 s (+4-10%).

### 4. CB_DEV vs threadgroup at d4-K2048 — NOT REFUTED (deliberately)
At d4-K2048 both arms are legal (16384+12288 = 28672 <= 32768, TG arm
today) with opposite traffic profiles: the TG arm bulk-loads 16 KB of
codebook per threadgroup (~1.15 GB/dispatch of L2-hot re-reads at OUT=1024)
then does O(1) threadgroup lookups; CB_DEV pays zero upfront and takes
dependent device loads instead.

This asks for a MEASUREMENT, not a kernel design change, so refuting its
feasibility is close to vacuous — the refuter was killed rather than spend
another ~70 minutes on it. It is already item 2 on KERNEL-COVERAGE's
remaining list. Just run the A/B; it could delete a whole branch.

## Recommended order

1. **NSUB=80 tail** — smallest, best-understood, and the only one that
   changes fleet COVERAGE rather than speed. The runtime already supports
   the format; this is a gate relaxation plus a numeric gate.
2. **Paired token tiles (RTILE)** — the refuter's own analysis says this is
   the lever that reaches the dominant term at d4-K8192.
3. **CB_DEV vs TG A/B** — one measurement, no design work, may delete code.
4. **OTILE** — real but second-order, and its refuter supplied the control
   (RTILE=16) needed to interpret a null. Do it after RTILE, not before.

## Method notes

- 342,927 tokens across 8 substantive workers; 24k-52k tokens and 30-86
  minutes EACH. GLM is extraordinarily verbose but the output is dense and
  code-anchored, not padding — the OTILE refutation alone caught an
  arithmetic inconsistency, an unpriced occupancy confound, and a wrong
  claim about the template list.
- The degeneration guard fired at least once (~1817 chars, verbatim loop).
- Serialization: one generation at a time on the gen-lock regardless of
  `--concurrency`, so wall-clock is the sum, not the max.
- This run predates the incremental-verdict checkpoint (scout 69ff8510), so
  its output file was reconstructed from `logs/swarm/transcripts`. Future
  runs checkpoint each verdict as it lands, and `scripts/swarm_peek.py`
  reads any run live.
