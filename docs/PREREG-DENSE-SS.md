# Pre-registration — the KL cost of VQ_DENSE_SS=1 (F136's 1.146x)

Registered 2026-09-18, before any scoring run. Noah's prior, stated in
session: "I think the initial look at it showed a serious cost."

## What the switch does

`_dense_ss` replaces a 32-step SERIAL reduction

    for (i=0..31) acc = fma(srow[b*32+i], simd_shuffle(gacc, i), acc);

with a TREE reduction: each lane scales its own group partial and one
`simd_sum` reduces the 32 in tree order. Same values, different summation
ORDER. model.py measured the divergence at max 1.00/4.00/2.00/7.00 ULP on
gate (N=1/5/10/20) and up to 8.00 ULP on down_proj, and shipped it OFF
pending "its own referee pass, on the dense line, at dense ULP."

F136 measured the speed: **1.146x whole-model decode** on 27B-VQ-3.9, and my
checksum moved, independently confirming the arithmetic changes.

## Predictions

* **P1.** |delta KL| < 5 mnats on all three corpora.
  Reasoning: 8 ULP of a float32 accumulator is ~1e-6 relative; logits are
  O(10), so an absolute perturbation ~1e-5, which is orders of magnitude
  below the 100-200 mnat scale these rungs sit at.
* **P2.** The shift is < 1% of the rung's own KL on every corpus.
* **P3, the non-obvious one and the reason this is worth running.** The SIGN
  may favour SS. A tree reduction's rounding error grows as O(log n) against
  a serial chain's O(n), so summing 32 partials in tree order is typically
  MORE accurate, not less. ULP divergence measures DISAGREEMENT between two
  roundings; it does not say the new one is the worse one, and nothing in
  model.py's note claims it does. **If SS lowers KL, the switch is not a
  quality cost at all and the "serious cost" reading is an artifact of
  treating ULP-vs-reference as ULP-of-error.**
* **P4.** |t| may EXCEED 2 while the delta is negligible. n = 12288 positions
  and the difference is deterministic, so the F118 gate's |t|>2 will detect
  an arbitrarily small real shift. **For this decision the MAGNITUDE governs,
  not significance.** Registered in advance so a significant-but-trivial
  result is not read as a cost.

FALSIFIER: if |delta| > 20 mnats on any corpus, P1-P3 are wrong together,
Noah's prior is right, and the switch stays off on quality grounds.

## Method

`vqlab kl-ladder`, three corpora at 12288 (q27_teacher_topk_{prose,code,lit}
_12k), artifact TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw.

VQ_DENSE_SS is process-global and kl_ladder's score_one inherits the parent
environment, so the two arms are TWO INVOCATIONS sharing one --per-pos-dir,
with rung names ss_off and ss_on. The paired delta is then computed with
kl_ladder's own `paired()` formula on the identical position arrays -- the
same test, applied across invocations rather than within one.


---

## CORRECTION to the method, 2026-09-18, before any result was read

**The first F137 run was VACUOUS and is discarded.** Both arms returned KL
identical to three decimals (prose 140.843 +/- 6.010, ppl 5.6544 on BOTH),
which is not a small effect -- it is the signature of both arms running the
same code, the F129 defect.

**Why.** kl_ladder scores at the chunk the cache records (512, and rightly so
-- F111: the metric is not chunk-invariant for recurrent-state families). But
`VQLinear.__call__` only takes the FUSED path when `N <= _fused_max_n`, and
for packed d4 that is `_DENSE_FUSED_MAX_N_BY_D[4] = 32`. At N=512 it falls
through to `_decode_matmul` (wdec + GEMM). **The SS kernels are never
reached.**

### The finding this exposes, which outranks the SS question

**THE KL GATE DOES NOT EXERCISE THE DECODE KERNELS.** On dense artifacts the
F118 referee scores through the PREFILL path. The fused decode kernels -- the
ones that serve token generation, and the ones VQ_DENSE_SS, VQ_D4_WALK and
the DEVX twins all modify -- are invisible to it. **Any numerics change
confined to the decode path passes the release gate untested**, not because
the gate is lax but because it never runs that code.

This is not hypothetical: F136 measured three decode-path switches on this
exact artifact, one of which (SS) is documented at up to 8 ULP. A release gate
that cannot see them is the gap, and `vqlab smoke`'s one-token generation is
the only thing that touches those kernels at all -- and it checks that a token
appears, not what it is.

Scope note: MoE artifacts may differ (different runtime, different max-N), and
this has NOT been checked for them. Worth its own pass.

### The corrected method

`VQ_DENSE_FUSED_MAX_N=1024` on BOTH arms forces the fused path at scoring N
(`_fused_max_n` selects `_DENSE_FUSED_MAX_N_PACKED` whenever that variable is
present). The arms then differ ONLY in the reduction, and since the SS change
is over NGRP and independent of N, this exercises the same arithmetic the
decode path runs.

CAVEAT to carry into the finding: forcing the fused path at N=512 is NOT the
shipped configuration. It is the only way to put the SS reduction under the
referee without building a chunk<=32 teacher cache, and the numbers it
produces price the REDUCTION, not the shipped scoring path.
