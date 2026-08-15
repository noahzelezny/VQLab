# E35 — Vector quantization for the 2-bit expert region (the last lever)

> **STATUS 2026-08-15 — HISTORICAL. This plan is DELIVERED; do not act on
> its "next steps".** M0 (35B quality), M0b (397B proxies), M1 (Metal
> kernel + runtime + exo, all five milestones) and M2 (the real shippable
> artifact) are all DONE. The shipped artifact is
> `rotlab--397B-tail3x3-vqK256codes`: **110.8 GiB, wikitext 2.7655 / code
> 2.6383** (beats spicyneuron 2.6bit on BOTH corpora at 9.8 GiB less),
> runs on a single 128 GB Mac at 30k context / ~20 tok/s decode, loads
> under stock unpatched mlx-lm.
> **Current truth lives in `EXPERIMENTS.md` (sections E35, M2, M2b, M2c,
> E36) and `M1_KERNEL_PLAN.md`.** Kept for the reasoning that got us here.

Scoped 2026-08-14, after E31-E34 closed every scalar-format lever. Noah's
call: willing to spend real time including kernel work ("sometimes things
are easier than we think").

## Why this attacks the actual wall

Every failed lever died against the same constraint: **scalar 2-bit affine
has 4 levels per weight, and any method needing representational headroom to
encode a correction loses it to rounding** (E34's chaotic-cascade result).
VQ removes that constraint instead of fighting it:

- Scalar 2-bit over a 4-weight group: 4^4 = **256 rigid grid patterns**.
- PQ d=4, K=1024: **1024 learned patterns**, same 10 bits of codes —
  positions placed where the joint weight distribution actually lives.
- Codebook cost is ~8 KiB amortized over millions of weights (~0 bpw);
  2.50 bpw codes + 0.25 bpw per-row scale = **2.75 bpw effective vs RTN's
  2.5** — matched-class, with a (d, K) dial to walk the curve exactly
  (d=4 K=256 → 2.0 bpw; d=8 K=65536/E8P → 2.0 bpw lattice-style).

This is the QuIP#/AQLM regime: the literature's 2-bit wins live HERE, not
in scalar grids. And E32's flatness finding is an asset now: lattice/VQ
methods *want* Gaussian-ish weights — the same property that made rotation
useless for scalar affine. `rotate_fuse.py` exists if incoherence helps
(measure with/without at M0; it's one flag).

## E34's standing constraint on the design

Fit codebooks in **pure weight space** (k-means on weight subvectors,
optionally per-row scaled). NO Hessian weighting, NO activation-aware
objectives at v0 — activation-fitted anything is 0-for-4 on this family.
If v0 quality is close-but-short, Hessian-diag weighting is a v1
experiment, run with the E34d ratio probe FIRST.

## The decoupling that makes this cheap to falsify

**Quality and kernel are independent questions.** A VQ artifact can be
decoded to a bf16 35B artifact (65G, disk is free) and scored by the
existing referee with NO kernel, NO exo change. The Metal work only begins
after quality is proven.

## Milestones

**M0 — quality go/no-go (35B, no kernel, ~half a day):**
`vq_fit.py`: k-means (k-means++ init, ~25 iters, mx on GPU) per
(layer, proj) over expert weight subvectors; per-row fp16 scale; d=4
K=1024 first. Decode → bf16 artifact → referee, both corpora, x2.
- Bar: beat RTN struct6 (wikitext 7.4285 / code 3.1653) at matched-or-fewer
  bytes. Variants worth one run each: ±rotation pre-pass, K=4096 (3bpw
  class), per-expert vs per-layer codebooks.
- If M0 fails → project dead for the price of an afternoon, and the
  publish-now path proceeds with a complete falsification log.

**M1 — Metal kernel prototype (1-2 days):**
`mx.fast.metal_kernel` (verified present, mlx 0.32.0 — JIT custom kernels
from Python, NO MLX fork needed). Kernel: fused LUT-matmul
`y[r] = Σ_g codebook[codes[r,g]] · x[g*d:(g+1)*d] * scale[r]` — never
materializes decoded weights. Correctness vs numpy decode, then tok/s vs
`gather_qmm` on one expert shape ([512,2048] and [2048,512]).
- Bar: ≥ 0.5x gather_qmm throughput (decode is memory-bound; codes are
  SMALLER than packed 2-bit affine, so ≥1x is plausible).
- Fallback if kernel perf stinks: decode-to-affine on load (VQ as a
  STORAGE format, dequantized into stock QuantizedSwitchLinear at load
  time — costs load latency + runtime bytes 2.5→~3.5bpw... wait, decode
  to bf16 costs 16bpw = dead; decode to affine 4-bit gs128 ≈ 4.25bpw —
  only viable if M0 quality survives a re-quantization to 4-bit, which it
  should: 4-bit RTN is near-lossless vs the VQ decode. Measure at M0.)

**M2 — SwitchGLU integration (1 day):**
`VQSwitchLinear` subclass in a patched `switch_layers.py` (exo already
runs env-gated patches on both nodes — E2 precedent; sources in
quantlab/patches/). Loader: recognize `*.codes` / `*.codebook` /
`*.vq_scales` tensors. Batched/sorted-expert path must match
QuantizedSwitchLinear's (`sorted_indices` variant). Smoke: 35B VQ artifact
generates coherently through exo on ONE node, then both.

**M3 — 397B (1 day + builds):**
Fit on the M4 (rig note in [[project_397b_activation_fitted_quant_fails]]),
same struct6 skeleton: VQ the 2-bit expert region only, keep 3-bit tail /
structure / routers as shipped. Cards on both nodes, vision graft, referee
x2 both corpora. Bar = the debut bar: wikitext ≤ 2.3614 at ≤ 165.6 GiB
(headroom says ~142-155 GiB lands the class win if M0's 35B margin
transfers even half).

## Risks, named

1. **k-means at scale**: 35B experts = ~10^9 subvectors/layer-proj. Fit on
   a subsample (10-100M), assign-all in chunks on GPU. Manageable.
2. **Kernel perf**: LUT gather per 4-weight group is random-access into an
   8 KiB table — fits threadgroup memory; should be fine, but simdgroup
   matmul tricks don't apply directly. This is the real unknown; hence the
   decode-to-affine fallback.
3. **exo two-node**: patched loader must land on BOTH checkouts (E2/E23
   lessons); single-box referee stays the instrument of record regardless.
4. **Curve honesty**: every quality claim at MATCHED BYTES vs the RTN point
   on the same skeleton, both corpora, x2 — per the standing rules.

## Order of operations

~~M0 is the only thing to build next~~ — *this was true on 08-14; every
milestone through M2 is now DELIVERED (see the status banner at the top).*
