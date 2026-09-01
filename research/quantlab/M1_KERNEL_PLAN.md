# M1 — the VQ Metal kernel (scoped 2026-08-15, while G cooks)

The gate between the E35 proxy numbers and anything a human can download.
Everything below is `mx.fast.metal_kernel` (present in our mlx 0.32.0):
custom Metal source JIT'd from Python, NO MLX fork.

## Format (per VQ'd tensor, replacing weight/scales/biases)

    codes     uint8 (K<=256) or uint16     [E, out, in/d]
    codebook  fp16                         [K, d]
    vq_scales bf16                         [E, out, in/64]

d=4/K=128 codes are 7 bits stored in 8 — the artifact stores 2.29 bpw
against the 2.00 analytic. Optional 4-bit packing later recovers it; ship
v1 unpacked (simplicity first, still 21 GiB under the RTN daily).
d=8/K=16384: uint16 codes = 2.00 bpw stored exactly. Convenient irony:
the PREMIUM geometry packs cleanly and the cheap one doesn't.

## The kernel: fused LUT-matmul (never materialize the weight)

Per output element: y[t,r] = Σ_g scale[e,r,g] · Σ_{j∈g} codebook[codes[e,r,j]] · x[t, j*d:(j+1)*d]

Two shapes matter and they want different kernels:
1. **Decode (M=1..few tokens)** — memory-bound. One threadgroup per row
   block; threads stream codes, gather codebook, FMA against x held in
   registers/threadgroup. VQ reads FEWER bytes than affine-2bit (2.0-2.29
   vs 2.5 bpw), so the roofline argues we can BEAT gather_qmm here.
2. **Prefill (M=hundreds+)** — compute-bound. Naive LUT-FMA will lose to
   simdgroup matmuls. Strategy: decode a [BR x BK] weight tile into
   threadgroup memory once, then simdgroup_matmul against many tokens —
   decode cost amortizes over M. This is also the fallback if (1)
   disappoints: decode-to-tile is the universal slow-path.

Codebook residency drives the design:
    d=4 K=128:   1 KB  -> threadgroup memory, trivially hot
    d=8 K=16384: 256 KB -> device memory, L2-resident (M3/M4 L2 is MBs)
Measure both; if K16384 decode stalls on codebook gathers, split the
codebook into banked halves per simdgroup.

## MoE gather (the real call signature)

`QuantizedSwitchLinear.__call__(x, indices, sorted_indices)` →
`mx.gather_qmm(..., rhs_indices=indices)`. Ours: `VQSwitchLinear` with the
same signature; kernel takes `indices` and resolves e per token. v1 ignores
`sorted_indices` (correct either way, just unexploited locality); v2 uses
the sorted path exo's runner emits.

## Milestones (each gated on the last, ~in order of risk retirement)

- **M1a (half day): correctness.** Single-expert kernel vs numpy decode:
  max |Δ| within fp16 accumulation noise on random + real-codes inputs.
  **DONE 2026-08-15** (m1a_kernel_test.py, on the M4): ~2e-7 rel with fp32
  accum on synthetic d4/K128 AND d8/K16384 AND real L0 codes (down_proj +
  gate_up); 3.5e-4 with fp16 output. Emit side: m1a_emit_codes.py
  (standalone — the fused fitter stayed untouched under the live E/F/G
  chain); real-tensor fits take 12-16 s at K128 on the M4.
- **M1b (half day): decode-shape benchmark** vs gather_qmm on the real
  sizes ([512,1024,4096] gate_up, [512,4096,1024] down; M=1,4,16).
  Bar: ≥0.5x gather_qmm tok/s (roofline says ≥1x is in reach).
  **DONE 2026-08-15** (m1b_bench.py, M4): best kernel (tg4: threadgroup
  codebook+x, uchar4 code loads, fp16 scales) = **0.66-0.88x** gather_qmm
  at M=4/16 on both tensors; M=1 is launch-overhead jitter. Bar met.
  TWO measurement gotchas recorded in the harness: (1) the honest baseline
  is the mlx_lm shape — shared x [T,1,1,IN], [T,k] indices, sorted_indices
  — the flat unsorted gather_qmm path degrades ~80x at N=128 and would
  have flattered us 45x; (2) uint16 codes double read traffic — K<=256
  must ship uint8. Remaining gap to 1x is threadgroup bank conflicts on
  random codebook gathers (v2 lever: banked/half4 codebook, simdgroup
  row reduction).
- **M1c (day): the tile/prefill path + gather indices.** Bar: prefill of
  8192 tokens within 2x of gather_qmm (prefill is a one-time cost per
  request; 2x there costs seconds, decode rate is what Noah feels).
  **DONE 2026-08-15** (m1c_prefill_bench.py, M4) — and the winner is NOT
  a fused tile kernel: **decode-to-dense + PADDED batched GEMM beats
  gather_qmm outright: 1.21x (down_proj), 1.28x (gate_up)** at 8192 tok.
  Decode kernel is ~8% of the cost (2-4 ms/128 experts); compact each
  expert's rows to a uniform cap (pad overhead 1.27x rows) and run one
  [E,cap,IN]@[E,IN,OUT] batched matmul per chunk. The row-batched
  gather_mm variant (65k M=1 matvecs) is the trap: 0.43x. Plan risk #1
  (simdgroup tiling) RETIRED — never needed. Integration note for M1d:
  the pad/compact step must run as mx ops on GPU (bench built it in
  numpy outside the timed region; exo's runner already sorts by expert).
- **M1d (day): VQSwitchLinear + loader + exo.** Patched switch_layers
  recognizing `*.codes`/`*.codebook`/`*.vq_scales`, BOTH node checkouts
  (E2/E23). Smoke: 35B VQ artifact generates coherently through exo.
  **SINGLE-BOX HALF DONE 2026-08-15**: vq_switch.py + patch_mlx_lm.py
  (hook walks attributes — tree_unflatten breaks on layers.0) +
  vq_35b_codes.py. rotlab-35B-vqK256codes = **10.1 GiB** (vs 62G bf16),
  generates coherently at **85 tok/s, 11 GB peak** on the M4 via
  `mlx_lm generate`. **EXO 2026-08-15**: live env mlx_lm patched on BOTH
  nodes (conda envs + dev venvs, 4 trees), codebook-replicate guard in
  auto_parallel.py both checkouts (committed in ~/exo on M3, applied on
  M4), builtin model card installed. Single-node exo placement SERVES in
  30s and generates coherently through /v1/chat/completions. Two-node
  ring blocked ONLY on hardware: jaccl dials errno-65 because the TB5
  cable is still unplugged from the GPTQ night (all 120Gb/s ports idle,
  only the 40Gb/s link live). **RESOLVED same day: TB5 replugged →
  2-node TensorShardMetadata instance loads, runners Ready, coherent +
  CORRECT generation through /v1/chat/completions. M1d CLOSED.**
  Second gotcha: M4 resolves models via PER-MODEL SYMLINKS in
  ~/.exo/models -> the SMB share; a new artifact needs its symlink or the
  node reports DownloadPending forever and the 2-node load never starts
  (runners stick at Connected — exactly what we saw).
- **M1e: end-to-end referee** on a real (small-bytes) VQ artifact vs its
  bf16 proxy score. Bar: PPL within noise of the proxy (same values, so
  any gap = kernel bug). THEN the artifact claims are real, sizes weighed
  not analytic, and the HF upload can happen.
  **DONE (35B) 2026-08-15, M4 referee, all runs bit-identical x2:**
    codes 7.0378 wiki / 3.0755 code; twin 7.0313 / 3.0750.
  Gap = +0.092% wiki / +0.016% code — NOT a layout bug (those are
  catastrophic, not 0.1%): the twin stores bf16(fp32 product) and runs
  dense-bf16 gather_mm, the codes path computes fp16-in/fp32-acc/fp16-out.
  Three dtype paths, sub-0.1% spread, quality claims move by nothing.
  Twin also confirms the E35 K256 record (7.1807 -> 7.0313 here; refit
  reseed + fp16 scales, slightly better). 397B claims will be re-measured
  on the real artifacts anyway, never carried from proxies.

- **M1f: d=8 kernels (E36/E37, 2026-08-15). DONE — and they are FAST.**
  `vq_switch.py` gained two d8 fused kernels plus a d-generic dense decode;
  dispatch is per MODULE on codebook shape, so one checkpoint can mix d4 and
  d8 tensors freely.
    - `_SRC_FUSED_D8` — codebook stays in DEVICE memory and rides L2. This was
      the open risk (K4096 = 64 KB fp16 vs Apple's 32 KB threadgroup memory,
      with random per-weight gathers, the pattern caches handle worst).
      **L2 residency holds**: 0.86-1.39x gather_qmm at M=1/4/16 on real 397B
      L0 down_proj codes — BETTER than the shipped d4 tg4 path (0.84-0.93x).
      Risk #2 in the list below is retired.
    - `_SRC_FUSED_D8_TG` — threadgroup variant for K<=1024 (16 KB): 0.99-1.12x.
      Kept as fallback; no reason to prefer it at K4096.
    - Prefill (decode-to-dense + padded batched GEMM, unchanged strategy):
      **1.19x** at 8192 tokens vs d4's 1.22x. No regression.
    - M1a correctness ~2e-7 fp32-accum on synthetic AND real d8 K4096 codes,
      covering both shipping paths (fused + dense decode).
  **But the artifact this was built for LOST** — see EXPERIMENTS.md E37. d8 for
  down_proj was a layer-0 measurement artifact; at layer 40 d8 is 32.9% WORSE
  than d4. The kernel is not the problem and the mixed-geometry machinery is
  referee-validated; the geometry CHOICE was wrong. C stays champion.

## Fit-side work M1 needs (small)

`vq_fit`/`vq_397b_fused` currently throw codes away. Add `--emit-codes`:
save codes/codebook/vq_scales per tensor alongside (or instead of) the
bf16 reconstruction. ~30 lines; do it with M1a so correctness tests run
against REAL codes from layer 0, not synthetic ones.

## Risks, ranked — ALL RETIRED 2026-08-15 (kept for the record)

1. ~~Prefill throughput (simdgroup tiling)~~ — **never needed.**
   decode-to-dense + PADDED batched GEMM BEAT gather_qmm (1.21-1.28x).
   The row-batched gather_mm variant is the trap (0.43x).
2. ~~K16384 codebook thrashing~~ — **MEASURED AND RETIRED 2026-08-15 (M1f).**
   The 64 KB d8 K4096 codebook stays L2-resident under random per-weight
   gathers and runs 0.86-1.39x gather_qmm — BEATING the d4 threadgroup path.
   Banking was never needed. (What killed d8 was quality, not residency:
   E37. The kernel is fine; the geometry choice was wrong.)
3. ~~exo two-node~~ — **done.** Both nodes patched, ring serves the VQ 35B.
   Two gotchas: numeric path parts break tree_unflatten (walk attributes),
   and the M4 resolves models via per-model symlinks in `~/.exo/models`.
4. NOT a risk: quality — **confirmed, and better than predicted.** The real
   artifact BEAT its own proxy (2.7655 vs 2.8197) because it ships fp16
   scales where proxies used bf16.

## What actually bit us instead (none of it was on this list)

- **The stored-vs-analytic bpw trap.** This doc's own note ("7 bits stored
  in 8") was never carried into the size claims. Codes are stored in whole
  bytes, so F/G are 110.8 GiB not ~100, and A/B/E need bit-packing to
  reach their quoted sizes at all. Full table in `EXPERIMENTS.md` M2.
- **A lazily-evaluated chunk loop holds every iteration's transients.**
  `_prefill` looked exactly like "the model is too big for this machine"
  (3.35 MB/token vs 0.059 MB/token of real KV cache). Fixed by `mx.eval`
  per chunk. This, not model size, was the context ceiling.
- **Benchmarks that measure the wrong path.** A monolithic prefill forward
  pass OOMs Metal at 8k where the chunked path mlx_lm actually uses runs
  30k fine; and tok/s that includes prefill understates decode ~10x.

## Where the frontier is now (see EXPERIMENTS.md E36)

~~Mixed geometry: gate/up d4 K256 + down d8 K4096~~ — **FALSIFIED (E37):
E36's d8 win was a layer-0 artifact; the real mixed build LOST both corpora
and was deleted.** Step 2 (bit-packing) WAS built and validated instead
(08-15/16): packed == unpacked bit-identically at 7/8/11 bits, 35B and 397B.

## FINAL STATE (08-16) — this plan is fully delivered and closed

Three real artifacts, all refereed x2 both corpora under stock mlx_lm:

  | | packed GiB | wikitext | code |
  |---|---|---|---|
  | F (d4 K128, 7-bit packed) | 100.1 | 3.1706 | 2.6988 |
  | C (d4 K256, byte-aligned) | 110.8 | 2.7655 | 2.6383 |
  | E (d4 K2048, 11-bit packed) | 142.8 | 2.3519 | 2.5987 |

One late kernel fix mattered: the original fused d4 kernel cached the
codebook as float4, so K2048 = 32 KB of threadgroup before x — Metal refuses
the load. `_SRC_FUSED_D4_BIGK` (half4 cache, value-identical) unblocked E.
Current truth: `EXPERIMENTS.md` (F REAL / E REAL / PACKED AT 397B SCALE).
