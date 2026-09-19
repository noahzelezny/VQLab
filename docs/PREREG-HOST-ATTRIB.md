# Pre-registration — separating F41's unattributed 12-13% host cost

*Written 2026-09-18 BEFORE any measurement, per instrument rule III.*

## The target

F41: the VQ module is 44% of prefill; the kernel body is 30 points, leaving
**14 points of host/dispatch/gather/cast**. `sorted_indices=True` accounts for
only 1.2 of those. The remaining ~12-13 points are named but **never
separated**: "the per-call numpy tile build in `_gemmseg_prefill`, the
`mx.array` uploads, kernel dispatch, and the broadcast/cast in `__call__`."

That 12-13% is about the size of the whole VQ-vs-affine parity gap, and four
swarm rounds of kernel work provably could not reach it.

## What reading the code changes about F41's list

F41 named "the per-call numpy tile build." It is MEMOIZED (`_memo_get` on
`("tiles", idx_sorted_np.tobytes(), E, _rt)`), as is the sort path
(`("sort", idx_np.tobytes(), k_rep)`). So the build itself is paid on a miss
only — but **every call pays `.tobytes()` to construct the key**, which is a
full byte copy of the routing array, then a dict hash over it.

Order of magnitude, 35B at 9000 tokens, top-8: `idx_flat` is ~72000 int32 =
~288 KB. Two memo keys per linear x 3 linears x 48 layers = ~288 key
constructions per forward, i.e. **~80 MB of byte copying and hashing per
forward pass** that exists purely to look up a cache.

This is a candidate F41 did not name because F41 did not read the memo.

## Predictions (recorded before running)

* **P1.** Memo-key construction (`.tobytes()` + dict hash) is the single
  largest host component, **> 2%** of total prefill wall time.
* **P2.** The argsort/permutation path shows **< 1.5%**, consistent with
  F41's measured 1.2% for `sorted_indices=True`.
* **P3.** `mx.array` uploads (tmeta, inv_mx, src) are **< 2%** combined.
* **P4.** The `.astype(mx.float16)` casts are **< 2%**.
* **P5.** `np.array(idx_flat, copy=False)` forces a device->host sync, and
  cProfile will NOT attribute its true cost — it will appear cheap while
  actually waiting on the GPU. Any conclusion drawn from its self-time is
  invalid.
* **P6.** The four named components will NOT sum to 12-13%. Some of the gap
  is per-dispatch overhead inside mlx that a Python profiler cannot see.

A falsified prediction is recorded as falsified, not reframed.

## Method

cProfile on a real prefill through a real artifact — attribution BEFORE
deletion arms, because deletion changes semantics and F41 already has one
INVALID ARM from exactly that (stubbing `np.argsort` to identity ran SLOWER;
it changed the kernel's memory access pattern and attributed nothing).

Profiling measures HOST time only, which is the correct scope: MLX is lazy,
so time around `mx.array()` is enqueue cost, not GPU work. The GPU side is
already attributed by F38/F39/F41.
