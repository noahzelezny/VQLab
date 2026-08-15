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

## Fit-side work M1 needs (small)

`vq_fit`/`vq_397b_fused` currently throw codes away. Add `--emit-codes`:
save codes/codebook/vq_scales per tensor alongside (or instead of) the
bf16 reconstruction. ~30 lines; do it with M1a so correctness tests run
against REAL codes from layer 0, not synthetic ones.

## Risks, ranked

1. Prefill throughput (simdgroup tiling is the fiddly Metal). Mitigation:
   decode-to-tile fallback is straightforward and correct.
2. K16384 codebook gathers thrashing. Mitigation: banked codebook; or ship
   accessibility artifact at d=4 K=128 (1 KB codebook) and keep d8 premium
   for a v2.
3. exo two-node integration (never the kernel itself — E2 history).
4. NOT a risk: quality. Proxies already carry the referee numbers; the
   kernel just has to reproduce them.
