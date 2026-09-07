# VQ kernel coverage matrix (2026-09-07)

The fleet now mixes codebook geometries inside single artifacts, so
coverage must be per-MODULE-GEOMETRY, not per-model. Columns are the
runtime's execution paths; ✅ = dedicated fast kernel, ⚠️ = works via
fallback (correct but slow), ❌ = missing/blocked.

## MoE (expert) modules — vq_switch

| geometry | rungs using it | decode small-N (fused LUT) | prefill fused (gemmseg v2) | prefill legacy (decode+GEMM) |
|---|---|---|---|---|
| d2 packed K512 b9 | 35B-4.6(part), 27B* | ✅ | ✅ **1.43x measured** | ✅ |
| d2 packed K1024 b10 | 35B-5.4, Flash-4.4/5.5(part) | ✅ | ✅ (gate fits; unbenched) | ✅ |
| d2 packed K2048 b11 | gemma-26b | ✅ ⚠️(8.7% divergence under investigation) | ✅ **1.35x measured** | ✅ |
| d2 unpacked K256 | Flash all rungs(part), 35B shared | ✅ | ❌ (needs BITS=0 arm — trivial) | ✅ |
| d4 packed K512 b9 | GLM-2.7/3.1(part), 397B-2.6 | ✅ | ❌ **r5 swarm in flight** | ✅ |
| d4 packed K2048 b11 | GLM(part), 397B-3.1, 35B-3.4, 35B-4.6(part), Flash-3.2(part) | ✅ | ❌ **r5 swarm in flight** | ✅ |
| d4 packed K8192 b13 | GLM-3.1/3.6(part), 35B-3.8 | ✅ (device cb) | ❌ needs device-cb variant (r5) | ✅ |
| d4 packed K16384 b14 | GLM-3.6(part) | ✅ (device cb) | ❌ needs device-cb variant (r5) | ✅ |
| d4 unpacked K256 | 397B-2.4 | ✅ | ❌ | ✅ |
| d8 packed K16384 b14 | 397B-2.2, Flash-2.1(part) | ✅ (device cb, simd) | ❌ needs device-cb variant (r5) | ✅ |

*27B rows in the audit are DENSE artifacts — gemmseg does not apply.

## Dense modules — vq_dense

| geometry | rungs | fused small-N | prefill (wdec + GEMM) |
|---|---|---|---|
| d2 packed b9 | 27B-4.8 | ✅ (tiled) | ✅ wdec **13x decode measured** |
| d2 unpacked | 27B-4.5 | ✅ (tiled) | ✅ (one-gather path, bit-frozen) |
| d4 packed b12 | 27B-3.9 | ✅ (tiled, device cb) | ✅ wdec (memory win; d4 speed note in handoff) |

## Embedding / PLE — separate family

| module | status |
|---|---|
| VQEmbedding / VQPLEEmbedding (NSUB=40, unpacked u16) | pure-gather path only; explicitly out of scope for wdec AND gemmseg; e4b PLE rung NO-SHIP (Metal defect) |

## Priority order for full ownership of the concept

1. **d4 threadgroup-cb gemmseg** (K512/K2048) — biggest fleet share:
   unlocks GLM + 397B-2.6/3.1 + completes the mixed 35B/Flash rungs.
2. **BITS=0 arm** — ~10 lines, completes every Flash rung's d2 share.
3. **device-cb gemmseg** (d4/d8 big-K) — 397B-2.2 flagship + GLM-3.6.
4. **gemma divergence** — correctness before coverage for that family.
5. PLE family — separate design (gather-indirected), lowest leverage.

r5 swarm (in flight) targets 1-3's designs; implementation follows the
v2 pattern: numeric gate rel<1e-3, resident-block 9k bench, score gate
with the prefill path FORCED (the scorer's default chunk never crosses
the threshold — see VQGEMM-BENCH-2026-09-07.md).
