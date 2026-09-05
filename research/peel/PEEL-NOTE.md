# Expert-peel experiment — 2026-09-05 (haiku-swarm + hand integration)

Peel heavy experts (count > VQ_PEEL_FACTOR × touched-median, default 2.0)
into SINGLETON groups; groups run the UNCHANGED _prefill loop body, so the
change is only which experts share a dispatch. expert_peel.diff applies to
vq_switch.py's _prefill (vq_peeled.py = truncated module carrying it).

Measured (M3, exo env, synthetic zipf routing a=1.1 ≈ 21x skew — HEAVIER
than the real 8.7x, so treat speedups as optimistic):
- OUT=2048 IN=5120 E=32 N=16k: **5.69x prefill wall** (168.2 → 29.6 ms),
  BIT-IDENTICAL to base.
- OUT=512 IN=1024 (small dims): 0.96x (decode-dominated, no win) and
  NOT bit-identical — ~3k elems differ, max 6.25e-2 (1-ULP class): mlx
  lowers small matmuls to a different kernel. Base itself IS bit-stable
  across its own _DECODE_CHUNK sizes (32 vs 16 identical), so
  repartitioning is bit-safe exactly when shapes stay in the steel-GEMM
  regime — which real projection dims do (empirically).

NOT SHIPPED. Before shipping: (1) real-routing measurement — apply the
sibling prefill_stats instrumentation, run a real Flash prefill, harvest
padding ratio; (2) referee bit-check on a real artifact per the lab law
(the small-dim divergence must be shown not to touch any real layer's
shapes); (3) decide default (VQ_PEEL_FACTOR=0 off vs 2.0 on).
Origin: 2026-09-05 swarm design doc survivor #1; probe/patch/harness by
a 3-agent haiku fan-out, integration + A/B by hand after the drafted
diff was malformed (small-model patch fragility — designs good, hunks bad).
