# MoE prefill attribution — measured (2026-09-05)

Question (from the swarm design round, logs/reviews/moe-prefill-design.md):
where do ~120 s of a 9k-token MoE prefill go on Scout? All 12 swarm
proposals hinged on this unmeasured attribution.

Probe: `scripts/probe_moe_prefill_attrib.py` — loads
`TheDrainFlorist--Qwen3.6-35B-A3B-VQ-4.6bpw` through its OWN bundled
model.py, instruments `_prefill` via the loaded module's globals, runs a
real 9000-token prompt (mlx_lm generate, default prefill chunking), REAL
router histograms per call. M3 Ultra, exo idle, repo .venv.

## Headline: the runtime is not the 120 s

| | wall | tok/s |
|---|---|---|
| 9k prefill + 1 tok, local runtime (unfenced) | **11.5 s** | **784** |
| same, phase-fenced (serialized, upper bounds) | 13.1 s | 689 |
| Scout's observed symptom | ~120 s | ~75 |

The bundled MoE VQ prefill path runs 9k tokens in 11.5 s on one box.
Scout's ~10x-slower experience is a SERVING-LAYER gap (exo cluster:
397B over the 2-box ring, chunked prefill + per-chunk ring sync, and/or
harness prefix-cache misses re-paying whole prompts), not this code.
Next measurement: the same 9k prompt through exo's API, cold and
prefix-warm.

## Real router facts (600 `_prefill` calls, E=256, top_k=8)

- touched experts/call: median **234 of 256** — near-total touch, so
  per-call decode amortizes over almost the whole expert set
- rows/call: median 16384 (2048-token chunk x top_k 8)
- skew max/mean: median **10.1x** (VQ-PF1's ~8.7x confirmed on 35B)
- **pad ratio: median 1.567 at the shipped chunk=32** — much worse than
  the 1.19 the VQ-PF1 comment reports for chunk=16. The padded GEMM does
  ~57% extra FLOPs at the shipped default.

## Fenced phase split (upper bounds; fencing kills pipelining)

| phase | s | % of instrumented |
|---|---|---|
| host map-building (numpy gmap/vmask) | 0.33 | 3.6% |
| `_decode_chunk` (expert weight decode) | 2.39 | 25.9% |
| padded `xp` activation gather | 2.24 | 24.2% |
| padded batched GEMM + per-chunk eval | 3.81 | 41.3% |
| unscramble (inv gather) | 0.46 | 5.0% |

Refutations this hands the next swarm round:
- **Host serialization is NOT the cost** (0.33 s / 3.6%) — the salvaged
  compute-frame suspicion is falsified at this scale/box.
- **Decode kernels are NOT dominant** (26% upper bound) — a 2x decode
  kernel win buys ~1 s of an 11.5 s wall.
- The real in-runtime levers, in order: padding waste (1.567 pad ->
  exact-cap sub-buckets could reclaim up to ~35% of GEMM time), the
  padded xp gather (24% — it materializes a full padded activation
  copy), then decode.
- But ALL in-runtime levers together bound at ~2x of an 11.5 s number.
  The 120 s lives in serving. Measure exo next.

## Shipped-bundle BUG found in passing (re-bundle list)

The 35B-4.6bpw bundle's `_get_kernel_spec` hardcodes the expert-FUSED
6-input signature (`x, eidx, codes, codebook, scales, dims` -> `y`) for
every specialized kernel — the arc-5 mis-binding the dev tree fixed with
`_kernel_sig`. Its `_decode_chunk` therefore CRASHES
(`Expected inputs to have size 6 but got size 5`) on any large-N prefill
with spec kernels on (the default). Decode/generation uses the fused path
and is unaffected, which is why smoke and tok/s look fine. Workaround:
`VQ_SPEC_KERNELS=0` (used by this probe). Fix: regenerate model.py from
the current dev tree (which shares `_kernel_sig`) — same class of fix as
the pending wdec re-bundle.

## Serving-layer measurement (same night, exo ring)

Attempted the same 9k prompt through exo's API:

- **Flash-Next-VQ-3.2bpw, ring-sharded across M3+M4 (auto-re-placed by
  the prewarm watcher): cold 9k prefill exceeded 20 MINUTES** (curl
  timeout at 1200 s, 0 bytes; exo logged CancelTask on disconnect — the
  request was processing, just glacially).
- Runner stdout explains it: `"Generating with a model that requires
  146552 MB which is close to the maximum recommended size of 86016 MB.
  This can be slow"` — repeated on every request. The instance is in
  mlx's OVERSUBSCRIBED regime: working set over the recommended wired
  limit → paging per forward → the 10-100x class. This, not kernels, is
  the shape of the user-visible ~120 s symptom (Noah's tuned placements
  set EXO_PREFILL_STEP_SIZE / EXO_MLX_MEM_LIMIT_GB per the vq-serving
  branch; the auto-re-placed instance may lack that tuning).
- **The shipped-bundle _prefill crash is PRODUCTION-CONFIRMED**: exo's
  log shows the 35B instance died in `model.py:2522 _decode_chunk`
  (the arc-5 mis-binding above) the moment the 9k request reached it —
  that is why the instance vanished mid-experiment. Any long prompt to
  the served 35B kills the instance until model.py is regenerated.

## Revised attribution for Scout's ~120 s

1. **Serving memory regime** (oversubscription/paging on the ring) —
   dominant, order 10-100x, fix is placement/limits/rung choice, zero
   kernel work.
2. **Ring chunked prefill + prefix-cache behavior** — unmeasured
   in isolation yet; the vq-serving branch knobs exist.
3. **In-runtime levers** (pad ratio 1.567, padded xp gather) — real but
   bounded ~2x of an 11.5 s baseline; worth doing after 1-2.
4. **Bundle crash** (35B) — not slowness but availability; re-bundle.
