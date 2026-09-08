# VQ kernel coverage matrix (2026-09-07, end of the fused-GEMM arc)

Coverage is per-MODULE-GEOMETRY, not per-model: the fleet mixes
geometries inside single artifacts. ✅ = dedicated fast kernel,
⚠️ = works via fallback (correct but slow), ❌ = missing.

## MoE (expert) modules — vq_switch

| geometry | rungs using it | decode small-N | prefill fused (gemmseg v2) |
|---|---|---|---|
| d2 packed K512 b9 | 35B-4.6 (part) | ✅ | ✅ **1.43x** |
| d2 packed K1024 b10 | 35B-5.4, Flash-4.4/5.5 (part) | ✅ | ✅ (gate fits) |
| d2 packed K2048 b11 | gemma-26b | ✅ | ✅ **1.35x** |
| d2 unpacked K256 | Flash all rungs (part), 35B shared | ✅ | ✅ BITS=0 arm |
| d4 packed K512 b9 | GLM-2.7/3.1 (part), 397B-2.6 | ✅ | ✅ threadgroup arm |
| d4 packed K2048 b11 | GLM (part), 397B-3.1, 35B-3.4, 35B-4.6 (part), Flash-3.2 (part) | ✅ | ✅ **1.30x** (pure-d4 rung) |
| d4 packed K8192 b13 | GLM-3.1/3.6 (part), 35B-3.8 | ✅ | ✅ **1.89x** CB_DEV arm |
| d4 packed K16384 b14 | GLM-3.6 (part) | ✅ | ✅ CB_DEV (unbenched at this K) |
| d4 unpacked K256 | 397B-2.4 | ✅ | ✅ BITS=0 arm |
| d8 packed K16384 b14 | **397B-2.2 (flagship)**, Flash-2.1 (part) | ✅ | ✅ **1.29x** (397B, ring) / **1.31x** (Flash-2.1, single-box) — PROMOTED to default 2026-09-07, see D8-BENCH-2026-09-07.md. ⚠️ 46 Flash-2.1 modules have NSUB=80 and stay legacy (gate wants a multiple of 32) |

Mixed-geometry artifacts fuse per module, so a rung with d2 and d4
halves gets both (35B-4.6 measured **1.65x** once d4 landed).

## Dense modules — vq_dense

| geometry | rungs | fused small-N | prefill |
|---|---|---|---|
| d2 packed b9 | 27B-4.8 | ✅ tiled | ✅ wdec **13x decode** |
| d2 unpacked | 27B-4.5 | ✅ tiled | ✅ one-gather (bit-frozen) |
| d4 packed b12 | 27B-3.9 | ✅ tiled, device cb | ✅ wdec (memory win) |

gemmseg is MoE-only by construction — dense prefill is wdec + GEMM.

## Embedding / PLE

VQEmbedding / VQPLEEmbedding (gather-indirected, NSUB=40): pure-gather
path only, out of scope for both wdec and gemmseg. e4b PLE rung remains
NO-SHIP (pre-existing Metal threadgroup defect).

## What remains

1. ~~**d8 bench**~~ — DONE 2026-09-07: 1.29x (397B) / 1.31x (Flash-2.1),
   promoted to default. See D8-BENCH-2026-09-07.md.
   **Successor item: the NSUB=80 tail.** 46 of Flash-2.1's 138 d8 modules
   have in=640 -> NSUB=80, and `gemmseg_fits` requires NSUB % 32 == 0, so
   they stay legacy. This is now the ONLY artifact in the fleet below 100%
   fused (98/144). Whether the kernel can take a ragged tail segment is
   unexamined. It was invisible until d8 was armed, because coverage
   previously reported all 138 as "arm unarmed".
2. **CB_DEV vs threadgroup where BOTH are legal** (d4 K2048): device is
   simpler and frees 16 KB — is it also faster? One measurement, needs
   a force-flag. Could delete a whole branch.
3. **The last ~15% to affine**: 1914 tok/s (d4-K8192) vs 2246 affine
   8-bit. Diminishing; residual ideas are bank-conflict padding on the
   wtT load, wider row tiles, dual-buffered x staging.
4. **PLE family** — separate design, lowest leverage.

Numbers and method: VQGEMM-BENCH-2026-09-07.md (all arms interleaved on
real artifacts with real router histograms; ratios reported, not
absolutes, because the machine drifts ~10% run-to-run).
