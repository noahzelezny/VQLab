# Fused segmented VQ-GEMM — overnight result (2026-09-07)

The r4 swarm designed it (21 proposals, 7 survivors, 4-frame convergence
on one architecture: logs/reviews/vqgemm-design-r4.md); implemented and
measured same night. Real 35B-A3B-4.6bpw through its own bundle, real
router, 9k-token prefill, M3 Ultra, cluster empty, 3 interleaved runs.

| arm | wall | tok/s | vs legacy |
|---|---|---|---|
| legacy (_decode_chunk + padded GEMM) | 8.4-8.5 s | 1064-1074 | 1.00x |
| v1 fused, scalar threadgroup MAC | 11.4 s | 793 | **0.74x — NEGATIVE** |
| **v2 fused, simdgroup matmul** | **5.8-6.0 s** | **1507-1547** | **1.43x** |
| affine 8-bit floor (stock, 09-06) | 4.0 s | 2246 | 2.11x |

VQ prefill moves from 44% to ~67% of the affine floor. The kernel:
dequant inside the tile loop (no fp16 weight materialization), one
dispatch per linear via per-tile (expert, row_start, nrows) metadata
over count-sorted rows (padding <32 rows/expert vs the padded-GEMM's
1.567x), Marlin-style decode-once-per-group into threadgroup, phase 3
as simdgroup_half8x8 matmuls with fp32 C tiles resident across the
whole group loop. wtT is stored transposed + pre-scaled at half.

v1 vs v2 is the lesson in one line: identical memory plan, identical
segmentation — scalar MAC loses 34%, simdgroup matmul wins 43%. The
ALU choreography, not the byte plan, was the whole game.

Gates: numeric rel<1e-3 vs legacy on skewed packed routing (both
variants; NOT bit-identical by design — reduction order differs; the
acceptance contract is in the source block). tests/test_vq_prefill_paths
covers both. DEFAULT OFF — VQ_MOE_FUSED_GEMM=2 opts in.

Score gate (2026-09-07 03:00): resident ppl, prefill path FORCED
(VQ_FUSED_MAX_N=1 — the scorer's default chunk=512 x top_k 8 = exactly
4096 pairs never crosses the threshold, so an unforced score measures
nothing; first run proved that by matching to 15 decimals):
legacy-prefill 5.4929, v2 5.4900 — a 0.05% shift, WITHIN the documented
benign float-reordering band (~0.16%, the chunk-8/16/32 shifts in the
runtime's own comments). Also noteworthy: standard published scoring
(chunk 512) rides the small-N fused path and would be UNCHANGED by v2.

NOT yet done (Noah's calls):
- Flash/GLM geometry check (gate currently requires d2/G64/packed —
  every current MoE rung qualifies, but only 35B was benched);
- promotion/publish.

Remaining gap to affine (~6.0 vs 4.0s): the non-MoE share (attention,
norms) plus residual kernel headroom (bank-conflict padding on wtT ld,
larger row tiles to amortize decode further, dual-buffer x staging).
Diminishing returns expected; measure before believing any of it.


## d4 extension (same day, r5 swarm design)

One parametric source now serves d2 and d4 (`D_BAKE` baked; phase 1
switches codebook entry width and halves-per-code, phase 3 untouched).
gemmseg_fits computes the E134 budget exactly: cb K*2*D + 3 tiles
<= 32KB, so d2 K<=5120 / d4 K<=2560 fuse and K8192+ falls THROUGH to
legacy (asserted in tests — never a kernel-LOAD failure).

Real 9k prefills, interleaved legacy/fused, M3 idle:

| artifact | geometry | legacy | fused | gain |
|---|---|---|---|---|
| 35B-3.4bpw | pure d4 K2048 (**no fusing before**) | 9.2 / 9.6 s | 7.2 / 7.2 s | **1.30x** |
| 35B-4.6bpw | mixed d2-K512 + d4-K2048 | 9.3 / 9.8 s | 5.8 / 5.7 s | **1.65x** |

The 4.6 number rose from this morning's 1.43x because its d4 half now
fuses too (both numbers are same-session interleaved; the morning's
legacy baseline measured 8.4-8.5 s vs 9.3-9.8 s now — run-to-run
machine drift of ~10%, which is exactly why arms must be interleaved
and why the ratio is the reportable quantity, not the absolute).

Score gate, 35B-3.4, prefill path FORCED: legacy 5.6820 vs fused 5.6840
(0.04%) — within the reordering-noise band, same as d2.

Still uncovered: d4/d8 big-K (8192/16384 — needs the device-codebook
arm; GLM-3.6, 397B-2.2, 35B-3.8) and unpacked d2 K256 (BITS=0 arm).
Both designed in logs/reviews/vqgemm-design-r5-d4d8.checkpoint.json.
