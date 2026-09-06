# MoE prefill optimizations — measured (2026-09-06)

The three r3-survivor runtime optimizations, implemented behind flags
(vq_switch dd6b828/f9a306d), A/B'd on the REAL 35B-4.6bpw through its
own bundle, real 9k-token prompt, real router, M3 Ultra, exo empty,
`probe_moe_prefill_attrib.py --no-instrument`. Two runs per arm.

| arm | wall (9k prefill + 1 tok) | verdict |
|---|---|---|
| legacy (all flags off) | 9.6 / 9.7 s | baseline |
| + gather fusion | 9.6 s | **NEUTRAL** |
| + vector-store decode | 10.2 / 9.7 s | neutral-to-noise |
| + exact per-expert GEMMs | 13.0 / 12.6 s | **-30%, NEGATIVE** |

## Conclusions

1. **Gather fusion** (`VQ_MOE_FUSE_GATHER`, default ON): bit-identical
   (uint16-gated) and wall-NEUTRAL at this scale. The 24% share the
   fenced attribution assigned to the xp gathers was a FENCING ARTIFACT
   — unfenced, those gathers overlap GPU work and cost ~no wall time.
   Stays ON (fewer allocations, zero risk), but claims no speedup.
2. **Exact-length per-expert GEMMs** (`VQ_MOE_EXACT_GEMM`, default OFF):
   **-30% CONFIRMED NEGATIVE.** ~230 small per-expert GEMMs lose to one
   padded batched GEMM even at the measured 1.567 pad ratio — padding
   FLOPs are cheaper than dispatch overhead + tile underutilization.
   This is VQ-PF1's row-batched-gather trap rhyming a third time: do
   not retry without a fundamentally different batching scheme. Flag
   kept (documented negative stays reproducible).
3. **Vector-store decode** (`VQ_DECODE_VEC`, default OFF): within noise
   (9.7-10.2 vs 9.6-9.7). The decode kernel was not store-bound enough
   to matter after wdec-era improvements. Stays OFF.
4. **Accidental real win, already shipped**: baseline moved 11.5 s ->
   9.6 s vs yesterday because yesterday's probe ran VQ_SPEC_KERNELS=0
   (workaround for the then-broken bundle). The sigfix re-bundle didn't
   just fix the crash — it turned spec kernels back on for the whole
   ladder: **~1.9 s (~17%) of 9k-prefill host overhead, live for every
   downloader since this morning's publish.**

## Lesson (again)

Refuters verify FEASIBILITY; only measurement decides PROFITABILITY.
All three survived adversarial review; measurement killed one, zeroed
another. And a fenced phase share is an upper bound on serialized cost,
not a prediction of pipelined wall — the 24%/41% attribution shares
were true and still didn't translate. The ~120 s user symptom remains
fully explained by serving-layer placement (see MOE-PREFILL-ATTRIB),
where the 60x lives; the runtime at 934 tok/s is not the problem.
