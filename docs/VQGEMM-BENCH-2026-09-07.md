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

NOT yet done (Noah's calls):
- score-identity run (ppl on a real eval through v2) before any default
  flip or artifact re-bundle carrying it ON;
- Flash/GLM geometry check (gate currently requires d2/G64/packed —
  every current MoE rung qualifies, but only 35B was benched);
- promotion/publish.

Remaining gap to affine (~6.0 vs 4.0s): the non-MoE share (attention,
norms) plus residual kernel headroom (bank-conflict padding on wtT ld,
larger row tiles to amortize decode further, dual-buffer x staging).
Diminishing returns expected; measure before believing any of it.
