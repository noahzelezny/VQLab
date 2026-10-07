# KernelProposals — prefill/decode kernel ideas with quoted source lines

Rule for this file (Claude's feedback, 2026-10-07): **every idea must quote the
current kernel lines it would change.** Round 2 skipped this and it cost us:
the quotes contradict idea 4 outright and soften idea 1's premise. Future
rounds append here with the same discipline.

Source of all quotes: `vqlab/src/vqlab/runtime/vq_switch.py` (the shipped
runtime; line numbers as of 2026-10-07, file is 5101 lines). MLX grounding:
`mlx/backend/metal/kernels/quantized.metal` + `quantized_utils.h` (fetched
2026-10-07). Web search was down that session (no BRAVE_SEARCH_API_KEY —
since fixed and verified working); direct fetches carried the round.

---

## Round 2 — affine-borrowing round (2026-10-07), corrected against source

### 1. vq_steel — vectorized decode staging + gather pipelining (REFRAMED)

**Original framing was wrong.** I wrote "gemmseg re-decodes the same codebook
slice per (row, group)". The source says otherwise — the decode is already
once per (out-row, group) and phase 3 already reuses it across 32 token rows:

```python
# vq_switch.py:3737-3743  (phase 1/2, _SRC_GEMMSEG2, d4 arm)
            const uint c = VQ_FETCH(j0 + q);
#elif D_BAKE == 4
                const half4 v = cb[c];
                wtT[q * 4][wr]     = (half)(s * (float)v.x);
                wtT[q * 4 + 1][wr] = (half)(s * (float)v.y);
                wtT[q * 4 + 2][wr] = (half)(s * (float)v.z);
                wtT[q * 4 + 3][wr] = (half)(s * (float)v.w);
```
`wtT` is `threadgroup half wtT[GROUP][32]` (vq_switch.py:3636) — the shared
decoded weight tile — and phase 3 mma's all 32 token rows against it
(vq_switch.py:3893-3905). So the BLOCK_M amortization steel has, we have.

**What we actually lack vs steel (the quotable deltas):**
- **(a) scalar strided half stores.** Four 2-byte stores per code, each
  landing in a *different* wtT row (`q*4`, `q*4+1`, …) at the same column
  `wr`. Steel stages tiles with vectorized loads/stores into padded TG
  buffers. Change: store `half4` chunks where the wtT layout permits
  (reindex wtT as `[wr][GROUP]` column-major so one code's D halves are
  contiguous → one `half4` store per code at d4, two at d8).
- **(b) per-code random device gather.** `cb[c]` (CB_DEV) is a dependent
  random device load per code — affine never pays this (its packed bytes
  are read coalesced and dequantized in registers). This is the structural
  VQ-vs-affine tax; round-1 #6 (group-major code layout) attacks its
  *coalescing*, nothing attacks its *latency* except idea 5 below.
- **(c) no gather pipelining.** PIPE exists but stops short — see idea 5.

Kill: microbench < +5% at 27B d4-K256, M=512/2048. Bit-exactness: (a) is
bit-exact (same elements, different layout); gate the composed kernel on KL.

### 2. VQ split-K for small-M prefill steps (unchanged, now grounded)

Borrow `affine_qvm_split_k` (spk 8, 32) / `qmm_t_splitk` from the MLX
instantiation list. K = NSUB·WPR·32 is huge; at step-512, M·N output is tiny.

Lines that would change — the serial group loop and the single-threadgroup
epilogue:
```python
# vq_switch.py:3714    for (int g = 0; g < NGRP; ++g) {          <- serial over K
# vq_switch.py:3893    for (int k8 = 0; k8 < G / 8; ++k8) {      <- phase 3 mma
# vq_switch.py:3990+   simdgroup_store(C0, &ybuf[0][...], 32);   <- one-TG epilogue
```
Split-K form: each threadgroup owns a g-slice, accumulates a partial, and a
second pass (or atomics) reduces partials into y.

**New grounding found this session (web search back up):** MLX PR #4628
("Keep split-K quantized-matmul partials in float32") and issue #4613
("qmm_splitk rounds partial sums to the input dtype") — stock MLX had a real
accuracy bug here: fp16/bf16 partial buffers rounded each partition's fp32
accumulator. **If we build VQ split-K, partials stay fp32 end-to-end.**

Kill: < +3% at M≤8. Untested class in the ledger (RTILE/OTILE were
tile-shape changes, not K-parallelism).

### 3. qmv_wide analog — batch 2–4 tokens per threadgroup in the small-N kernel

The step-512 small-N fused kernel stages ONE token's x per threadgroup:
```python
# vq_switch.py:637-641  (_SRC_FUSED_D4_BIGK)
    threadgroup half4 cb[MAX_K];
    threadgroup half4 xs[MAX_NSUB];
    ...
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4(...);
```
One `t` per threadgroup (`uint t = thread_position_in_grid.y`). The codebook
staging (`cb`, K·8 B at d4) is paid once per token. A "wide" variant stages
`vecs_per_tg` 2..4 x-rows and reuses the same staged codebook across them —
exactly MLX's `affine_qmv_wide` (`vecs_per_tg` 2..5, `k_lanes` 8). Distinct
from #2 (M-batching inside the threadgroup, not K-parallelism).

Caveat from the ledger: RTILE=64 on the *gemmseg* path measured 0.75x at
d4-K2048 — but that was the CB_DEV threadgroup-cap regime, not this kernel.
Kill: < +2% at step-512 shapes.

### 4. ~~Pack-time pre-scaled codebook~~ → **4'. accumulator-side scaling** (REPLACED — source kills the original)

**The original idea dies at the source.** I proposed folding `scale_g` into
each codebook entry at pack time. The scale is not a codebook property —
it is per (out-row, group):
```python
# vq_switch.py:3704-3706  (_SRC_GEMMSEG2)
    const device half* srow_base = scales
        + (size_t)e * OUT * NGRP + (size_t)(o0 + wr) * NGRP;
...
            const float s = (float)srow_w[g];
```
One shared `cb[c]` serves all 32 out-rows; `s` differs per out-row. Nothing
to fold.

**4'. The feasible form: scale the accumulator, not the weights — but Claude's
review (2026-10-07) shows it is NOT the cheap win it first looked like.**
y = Σ_g s[row,g]·dot_g, so the per-element `s *` multiply + `(half)` round in
phase 1 (quoted above, 4 stores per code) can in principle move to phase 3:
after each group's mma, scale that group's contribution by a per-out-column
broadcast and add it in. Phase 1 becomes a pure gather-copy: `wtT[...] = v.x`
with no arithmetic.

**Claude's pushback (2026-10-07), accepted in full:** because `s` varies per
out-row *and* per group, the per-group mma cannot chain into the single
accumulator. Each group's mma must land in its own temporary result, get
scaled column-by-column (broadcast `s[row,g]` over the token axis), and then
be added into C — instead of chaining into one accumulator. The trade: ~1024
fewer scale-multiplies per group at the quoted shapes (2048 (out,code) pairs
→ 1024 (out,tok) accumulator elements), bought with an extra scale-and-add on
every group, a second accumulator tile of register pressure, and per-group
temp management. **Could easily be slower — this is now a measured question,
not an assumed win.** The "closer to the fp32 reference" claim is likewise
plausible but unproven: kernel-truth exists precisely to measure that against
an exact float64 reference, BEFORE any KL run.

Kill ladder (in order; each step gates the next):
1. Speed microbench first. Kill: < +2% (or flat) at 27B d4-K256, M=512/2048,
   n≥3, reported as a ratio.
2. Only if speed survives: kernel-truth vs the exact float64 reference —
   report the signed relerr delta vs the current kernel (does it land closer,
   as claimed?).
3. Only if both survive: KL/ppl gate per law III.12.

### 5. Double-buffered code tiles — pipelining the *gather* (REFINED — PIPE already exists)

PIPE already prefetches the *independent* loads (codes + scale):
```python
# vq_switch.py:3717-3721
#if PIPE
    // Prefetch pipeline (kernel-body campaign arm 1.5, salvaged design):
    // g+1's code words + scale are INDEPENDENT device loads issued before
    // the mma, so they fly during the matmul; after the barrier only the
    // dependent codebook gather remains.
```
The un-pipelined remainder is exactly the dependent `cb[c]` gather. Change:
double-buffer `wtT` (two G×32 tiles, ping-pong) so group g+1's decode
(including its gathers) executes during group g's phase-3 mma, behind the
existing barrier structure. Costs +4 KB threadgroup at d4 (fits the 32 KB
cap only in non-CB_DEV shapes — scope to those, or shrink via idea 4'
removing nothing... wtT is the tile itself; the second buffer is the cost).
Kill: < +2%. Composes with round-1 #6 (that fixes transaction efficiency,
this hides latency).

### 6. 128-bit code loads — AUDITED, confirmed scalar (bonus, cheap, bit-exact)

Confirmed from source — codes are fetched as scalar words, one load per
code, and straddling codes pay a second load:
```python
# vq_switch.py:592-599  (_PACK_FETCH)
    #define VQ_CODE(crow, j) ( ( ((crow)[VQ_W(j)] >> VQ_SH(j)) \
        | ((VQ_SH(j) + BITS > 32) ? ((crow)[VQ_W(j) + 1] << (32 - VQ_SH(j))) : 0u) \
        ) & VQ_MASK )
# vq_switch.py:3700-3702  (BITS==0 arm — one scalar uint PER CODE)
#if BITS == 0
    #define VQ_FETCH(j) ((uint)wrow_codes[j])
```
And the access pattern is uncoalesced by construction: codes are row-major
per out-row (`codes + e*OUT*WPR + (o0+wr)*WPR`, vq_switch.py:3698-3699), so
the 128 threads of a threadgroup read 32 *different* rows. The audit's
actionable form is therefore round-1 #6's group-major layout (codes as
`[NGRP][OUT][WPR]` so a tile's 32 rows are contiguous per group) — the
source confirms both the scalar-fetch cost and the uncoalesced pattern.
Also worth checking: whether `VQ_CODE`'s straddle second-load can be
eliminated by aligning pack boundaries to code width at pack time
(pack-only, bit-exact).

---

## Sequencing (Claude's review, 2026-10-07 — the clean order)

Round 2's closing sequence was muddled (it self-corrected mid-sentence).
Claude's order, which this file adopts:

1. **#6 / group-major code layout** — phase-1 microbench. Bit-exact, no
   quality gate. Kill: < +3% prefill at 27B d4-K256, M=512/2048 (threshold
   proposed 2026-10-07 — confirm before running).
2. **#1(a) half4 store reindex of `wtT`** — bit-exact. Kill: < +2%, same
   shapes, n≥3.
3. **4′ accumulator-side scaling** — SPEED FIRST (kill < +2% or flat), then
   kernel-truth vs the float64 reference, then KL only if both survive.
4. **#2 split-K / #3 qmv_wide** — only if the small-M regime matters for the
   serving target.

Every step: n≥3, one process per arm, never on a contended box, and quote a
RATIO between arms from the same session, never an absolute (AGENTS.md law
IV speed rule; the decode instrument is bimodal at ~100 GiB).

---

## Provenance & rules

- All ledger numbers cited are pre-existing measurements (F/E entries in
  FINDINGS.md); nothing in this file is measured yet.
- Quality-gated changes (4') decide on KL/ppl, never relerr (law III.12).
- Rejected-list still stands: RTILE/OTILE, DBUF, SG2X2, wtT bank padding
  beyond WT_PAD=8, GWALK, walkers on prefill, exact per-expert GEMMs,
  DEVX/U4/R2, XKREP-for-prefill, hot-prefix codebook,
  simd_shuffle-resident codebook, chunk=16.
## Standing outranking items (outrank every idea above)

- **F188's 2.5x Knurlogic serving gap is UNVERIFIED — and it outranks all six
  ideas.** The ledger entry itself flags the flaw. Quoted verbatim from
  `lab/FINDINGS-LOG.md` F188 (2026-09-30):

  > MEASURED:   Knurlogic M4-only 2.7: 98/97 tok/s; same 2.7 split M4+M3
  > pipeline: 154/154 tok/s; 3.6 split ~69 tok/s. decode-ladder M4 2.7
  > baseline 8.22/8.45 s (249/242 tok/s, drift 2.7%), attention deleted
  > 5.22 s (393 tok/s), shared expert deleted 8.03 s; VQ arm VOID (0.004 s:
  > the stub collapsed the graph, not a timing). Microbench 42 MoE layers x
  > 2048 tokens: K512 2.65 s, K2048 2.60, K8192 2.68, K16384 2.82.

  > The largest lever found is the ~2.5x gap between Knurlogic's prefill and
  > a plain forward of the same bundle on the same Mac (prompt lengths
  > differ, 2734 vs 2048, so re-measure at equal length before quoting).

  The two arms used DIFFERENT prompt lengths (2734 Knurlogic TTFT vs 2048
  decode-ladder), so the 2.5x is not length-controlled. F190 (2026-10-01)
  later ran the Knurlogic pipeline at equal length (prompt 2048,
  KNURLOGIC_PREFILL_CHUNK=2048) but as a ratio vs the 4-bit affine
  reference — not the Knurlogic-vs-plain-forward pair — so the gap is still
  formally unverified. Verification plan: Knurlogic TTFT (max_tokens 1) at a
  2048-token prompt, n≥3, AND decode-ladder plain-forward at BOTH 2048 and
  2734, both directions, same session, ratio quoted.

- **Why it outranks the kernels:** F191 (2026-10-01) partitioned the GLM
  prefill — MoE 61.8% — and measured "synthetic VQ expert kernels at these
  shapes: ~63 ms/layer vs ~137 ms/layer measured MoE stage". The VQ kernels
  are under half the MoE stage, so a PERFECT kernel roughly halves the MoE
  stage → ~31% of prefill best case. The 2.5x serving gap, if real, is
  larger than any kernel win available. Also per F191: rerun at reps 2
  before quoting single layers (drift +10.5% at best-of-1).

- **u8view (+33% decode, bit-exact, E81) is still unshipped** in the
  published bundles. A rebundle is a write to another session's experiment
  (AGENTS.md, "Two agents, one artifact root"): announce the fleet-wide
  write first, pin + smoke (`vqlab smoke --max-tokens 8` AND
  `vqlab vision-smoke`) ONE artifact per family, gate ONE per family before
  the second. F154 is the cost of skipping that gate.

---

## Provenance & rules