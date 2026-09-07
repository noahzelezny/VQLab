# gemma-26b 8.7% two-path ppl divergence — forensics

**Date:** 2026-09-07
**Artifact:** `~/.exo/models/TheDrainFlorist--gemma-4-26b-a4b-it-VQ-6.2bpw`
**Reported:** `score_ppl_resident.py` ppl 809.87 (default chunk → fused arm) vs
745.41 (`VQ_FUSED_MAX_N=1` → prefill arm). Δ = 8.7% in ppl = **0.0829 nats**
mean NLL. On Qwen3.6-35B-A3B-VQ-4.6bpw (d2 K512 b9) the same two arms agree to
0.1%.

**Verdict: the packed-d2 kernels are exonerated.** No geometry-specific defect
exists at K=2048/bits=11. The two arms differ by exactly the fp16 rounding
budget they differ by on every other geometry, gemma's included. The ppl gap is
almost certainly metric amplification, not a numeric bug — see §4.

Runtime code was not modified.

---

## 1. Artifact geometry (from config.json / index, no weights loaded)

`config.json` → `vq_modules`, 90 entries, exactly two shapes:

| module | dim | K | pack_bits | group | IN | OUT | E | NSUB | NGRP | WPR |
|---|---|---|---|---|---|---|---|---|---|---|
| `layers.N.experts.switch_glu.{gate,up}_proj` (60) | 2 | 2048 | 11 | 64 | 2816 | 704 | 128 | 1408 | 44 | 484 |
| `layers.N.experts.switch_glu.down_proj` (30) | 2 | 2048 | 11 | 64 | 704 | 2816 | 128 | 352 | 11 | 121 |

`text_config`: 30 layers, `num_experts` 128, `top_k_experts` 8,
`hidden_size` 2816, `moe_intermediate_size` 704,
`final_logit_softcapping` 30.0, `hidden_size_per_layer_input` **0**.

**No VQPLEEmbedding shards in this artifact** (`hidden_size_per_layer_input: 0`;
no PLE tensors in `model.safetensors.index.json` — only 180 `codes`/`codebook`
entries, all under `experts.switch_glu`). The gemma-4-e4b PLE defect cannot be
in play here. Everything else in the checkpoint is affine (206 explicit 8-bit
entries; top-level default `bits: 2, mode: affine`), and is **identical between
the two arms** — the arms branch nowhere except inside `VQSwitchLinear.__call__`.

## 2. Dispatch-chain map

`VQSwitchLinear.__call__` (vq_switch.py:3542) computes `N = top_k * tokens` and
branches on `N <= VQ_FUSED_MAX_N` (default **4096**).

At the scorer's default chunk 512 with top_k 8, `N = 4096` — **exactly on the
boundary**, so the default arm is `_fused`. (Chunk 513 would flip it.)

### Arm A — `_fused` → `_fused_resolve` (2403 / 2428)
- `pack_bits=11 ≠ 0`, so the u8-view and d2-u32-view reinterprets are skipped.
- `D == 2` branch (2491): `vq_fused_packed11_d2` / `_SRC_FUSED_PACKED_D2`.
  **There is no `_d2_tg_fits`-style capacity check here** (unlike `D == 4`,
  which falls back to `_d4_tg_fits` → device-cb). Checked by hand: the d2 kernel
  allocates `half2 cb[MAX_K]` + `half2 xs[MAX_NSUB]` = `(K + NSUB) * 4` B.
  gate/up: `(2048 + 1408) * 4 = 13,824 B`; down: `(2048 + 352) * 4 = 9,600 B`.
  Both well under Apple's 32,768 B cap, so the missing check is latent, not
  live, for this artifact. (It *would* bite at e.g. K=4096 with NSUB≥4096.)
- WPR shape assert (2547) passes: 484 / 121 both equal `ceil(NSUB/32)*bits`.
- Template `[T, MAX_K=2048, MAX_NSUB, BITS=11]`; grid
  `(ceil(OUT/256)*256, N, 1)`, threadgroup `(256,1,1)`.
- Code fetch is `_PACK_FETCH`'s `VQ_CODE` macro — dim-agnostic, block-of-32,
  straddle guarded by `(sh + BITS > 32)` ternary. **No uint8/uint16 typing is
  involved anywhere on the packed path**: codes are `uint32` words for every
  `bits`. The 2026-08-18 uint16 class of bug is structurally absent here.
- Differences K=2048/b11 vs K=512/b9 on this path: only `MAX_K` (8 KB vs 2 KB
  threadgroup) and `BITS` (11 vs 9) as template constants. **No branch, no cap,
  no dtype, no `_EXPERT_SIMD_MAX_N` gate changes** — `simd_rows` stays `None`
  for `D == 2` at every K.

### Arm B — `_prefill` (3331)
- **The premise that this is "the legacy decode + padded GEMM path" is stale.**
  `_prefill` first tries `gemmseg_fits(D=2, K=2048, G=64, bits=11, IN)`, which
  returns **True** here (`K*4 + 2*32*64*2 = 16,384 ≤ 30 KB`; `NSUB%32==0`;
  `IN%64==0`), so the arm actually runs `_gemmseg_prefill` →
  `vq_gemmseg2_packed11_d2` (`VQ_MOE_FUSED_GEMM` defaults to `"2"`, promoted in
  a01bafc). The `__call__` fused-gather form supplies `xsrc`/`src_rows`, which
  is the precondition.
- This does **not** matter for the finding: measured below, on this geometry
  gemmseg-v2 output is **bit-identical** to the legacy `_decode_chunk` + padded
  batched GEMM. So arm B is well-characterised either way.
- `_decode_chunk` (3247): `pack_bits` → `vq_decode_packed11`, D-generic,
  `in_features` passed explicitly, no dtype restriction on the packed path.

## 3. Synthetic matrix (measured, this box, ~10 MB tensors, no weights)

Synthetic modules at the artifact's exact `(OUT, IN, G)` with `E=8`, random
codes, fp16 codebook/scales, fp16 activations. Reference = numpy fp32 built on
`vqlab.vq_pack.unpack` → gather → per-group dot → scale. Error normalised by the
reference output's standard deviation.

### 3a. Every path vs the numpy fp32 reference (max relative to |ref|max)

| shape | K | bits | N | fused | legacy decode+GEMM | gemmseg2 | gemmseg1 |
|---|---|---|---|---|---|---|---|
| up (704×2816) | 512 | 9 | 8/64/512 | 3.1e-4 / 2.5e-4 / 2.3e-4 | 3.3e-4 / 3.8e-4 / 3.5e-4 | = legacy | = fused |
| up (704×2816) | 1024 | 10 | 8/64/512 | 3.0e-4 / 2.6e-4 / 2.4e-4 | 3.8e-4 / 3.5e-4 / 3.6e-4 | = legacy | = fused |
| up (704×2816) | **2048** | **11** | 8/64/512 | 2.9e-4 / 2.6e-4 / 2.4e-4 | 4.0e-4 / 3.8e-4 / 3.9e-4 | = legacy | = fused |
| down (2816×704) | 512 | 9 | 8/64/512 | 2.7e-4 / 2.2e-4 / 4.4e-4 | 3.7e-4 / 3.5e-4 / 4.4e-4 | = legacy | = fused |
| down (2816×704) | 1024 | 10 | 8/64/512 | 2.4e-4 / 2.5e-4 / 3.9e-4 | 3.7e-4 / 4.0e-4 / 3.9e-4 | = legacy | = fused |
| down (2816×704) | **2048** | **11** | 8/64/512 | 2.5e-4 / 2.5e-4 / 4.3e-4 | 3.8e-4 / 3.9e-4 / 4.3e-4 | = legacy | = fused |
| qwen35 up (1024×2048) | 512/1024/2048 | 9/10/11 | 8/64/512 | 2.8–3.4e-4 | 3.3–4.4e-4 | = legacy | = fused |

Every cell is fp16 rounding. **No divergence anywhere.** N was swept across the
`VQ_FUSED_MAX_N` boundary by dispatching each path explicitly, so the boundary
itself is covered.

### 3b. Path vs path, the number that matters (relative to output σ)

| shape | K | bits | N | fused−legacy max | fused−legacy **mean** | fused−gemmseg2 | fused−ref mean | legacy−ref mean |
|---|---|---|---|---|---|---|---|---|
| gemma up | 512 | 9 | 512 | 3.89e-3 | **2.75e-4** | identical to fused−legacy | 2.3e-4 | 3.8e-4 |
| gemma up | **2048** | **11** | 512 | 3.78e-3 | **2.75e-4** | ″ | 2.3e-4 | 3.8e-4 |
| gemma down | 512 | 9 | 512 | 3.87e-3 | **2.74e-4** | ″ | 2.3e-4 | 3.7e-4 |
| gemma down | **2048** | **11** | 512 | 3.76e-3 | **2.73e-4** | ″ | 2.3e-4 | 3.7e-4 |
| qwen35 up | 512 | 9 | 512 | 4.54e-3 | 2.74e-4 | ″ | 2.3e-4 | 3.8e-4 |
| qwen35 up | **2048** | **11** | 512 | 4.51e-3 | 2.75e-4 | ″ | 2.3e-4 | 3.8e-4 |
| qwen35 down | 512 | 9 | 512 | 6.41e-3 | 2.74e-4 | ″ | 2.3e-4 | 3.8e-4 |
| qwen35 down | **2048** | **11** | 512 | 6.37e-3 | 2.74e-4 | ″ | 2.3e-4 | 3.8e-4 |

**The three load-bearing facts:**

1. The fused-vs-prefill mean divergence is **2.7e-4 σ, flat across K ∈ {512,
   1024, 2048}, bits ∈ {9,10,11}, N, and both gemma and qwen shapes.** The
   artifact family that "agrees within 0.1%" has the *same* per-linear
   divergence as the one that shows 8.7%. Whatever the ppl gap is, it is not
   produced by a K=2048/bits=11 kernel.
2. The fused arm is **closer** to the fp32 reference (2.3e-4) than the prefill
   arm (3.8e-4) on every row — expected: `_fused` accumulates one row in fp32
   directly from codes, while decode+GEMM materialises fp16 weights first and
   then rounds again in the GEMM. So the gap is genuine fp16 disagreement in the
   direction physics predicts, with the *default* arm the more accurate one.
3. `vq_gemmseg2_packed11_d2` is **bit-identical** (`np.array_equal` True, max
   abs diff 0.0) to the legacy decode+GEMM on this geometry, verified by
   flipping `_FUSED_GEMM`. The gemmseg promotion is not implicated.

## 4. Root-cause hypotheses, ranked

**H1 — The metric, not the runtime (strongly favoured).**
The artifact's own model card states it outright: *"Perplexity is NOT reported
here, and that is deliberate. Raw loglikelihood is invalid on the gemma-4-it
family — the RL-sharpened distribution collapses, so wikitext ppl reads ~100–700
while the model generates fluent prose. Verified independently against HF
transformers on unquantized bf16."* Both measured numbers, 745 and 810, sit
inside that documented invalid band. On a collapsed, saturated distribution
(plus `final_logit_softcapping: 30.0`), mean NLL is tail-dominated: a handful of
positions where the true token sits at ~1e-9 probability carry most of the sum,
and a 1-ULP fp16 shift reorders them. 0.083 nats over 2048 positions is ~170
total nats — a few dozen tail tokens moving is enough. The 35B control is not a
control for the kernel; it is a control for *metric sensitivity*, and it is
confounded: 35B's mean NLL is ~1.70 (ppl 5.49), a regime where the same 2.7e-4
per-linear noise cannot move ppl by more than a fraction of a percent. Same
kernel noise, wildly different metric gain.
*Evidence:* §3b (identical per-linear divergence on both families), the card,
`final_logit_softcapping`.

**H2 — Real amplification through 30 layers of a genuinely more chaotic model
(possible, and not exclusive of H1).**
gemma-26b routes top_k 8 of 128 experts on a narrow 704-wide expert. A 2.7e-4
perturbation on the router's input can flip a *routing decision*, which is a
discrete change, not a rounding one — and a flipped expert at layer 3 is a large
downstream delta. Qwen3.6-35B has fewer/wider experts and less top-k churn. This
would make the ppl gap "real" in the sense of two genuinely different forward
passes, while still meaning neither path is *wrong*. Distinguishable from H1 by
whether the ΔNLL is tail-concentrated (H1) or broadly distributed (H2).
*Evidence:* geometry (128 experts, top_k 8, moe_intermediate 704); no direct
measurement yet.

**H3 — A defect in the packed-d2 K2048/bits11 fused kernel (FALSIFIED).**
Ruled out by §3a/§3b: exact-shape synthetic agreement with a `vq_pack.unpack`
numpy reference at fp16 rounding, at both K=512 and K=2048, on both module
shapes, across N. No uint8/uint16 typing exists on the packed path; no branch,
threadgroup cap, or simd gate differs between K=512 and K=2048 for D=2.

**H4 — A gemma-specific VQ module (VPLEEmbedding / vision shards) (FALSIFIED
for this artifact).** `hidden_size_per_layer_input: 0`; the index contains no
PLE or vision VQ tensors — all 180 VQ tensors are expert `switch_glu`. And any
such module would be arm-invariant anyway: nothing outside
`VQSwitchLinear.__call__` reads `VQ_FUSED_MAX_N`.

**Latent (unrelated to this divergence, worth a ratchet):** `_fused_resolve`'s
`D == 2` packed branch has **no threadgroup-capacity fallback**, where `D == 4`
has `_d4_tg_fits` → `_SRC_FUSED_PACKED_D4_DEVCB`. `(K + NSUB) * 4 > 32768`
would fail to *load* the kernel (a loud XPC error, not silent garbage), but it
is an unguarded cliff for a future d2 artifact at K≥4096 or a very wide IN.

## 5. The ONE discriminating full-model experiment

Run when the GPU frees. **Establish the ppl noise floor of this artifact before
attributing anything to the path split**, then split the tail:

```bash
M=~/.exo/models/TheDrainFlorist--gemma-4-26b-a4b-it-VQ-6.2bpw
# Control: SAME arm, two chunk sizes, both under VQ_FUSED_MAX_N=4096
#   (chunk 512 -> N=4096 fused; chunk 256 -> N=2048 fused).
# Different GEMM/dispatch tiling, identical code path.
python scripts/score_ppl_resident.py --model $M --max-tokens 2048 --chunk 512
python scripts/score_ppl_resident.py --model $M --max-tokens 2048 --chunk 256
```

**Decision rule.** If the two *same-arm* runs differ by anything of the order of
a few percent, the 8.7% is inside this artifact's own metric noise floor, H1 is
confirmed, and there is nothing to fix — record it and move on. If they agree to
<0.5% while the fused-vs-prefill split stays at 8.7%, the split is real and H2 is
next: re-run both arms with per-token NLL dumped, and check whether the 0.083-nat
mean gap is carried by the top ~1% of positions (tail reordering, still benign)
or spread across the corpus (a genuine forward-pass difference — then instrument
router top-8 selections per layer and count flips between arms).

Secondary confirmation either way, and the metric the artifact actually ships on:
score **KL-to-bf16** under both arms. If KL agrees between arms while ppl does
not, both paths compute the same model and only the collapsed ppl estimator
disagrees.

---

*Probe scripts were scratch and were deleted; the matrix above is reproducible
from §3's description in ~2 minutes. Do not trust these numbers as current —
re-run the probe.*
