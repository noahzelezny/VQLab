# The dense 27B VQ artifacts are broken and slow — three distinct causes

**Status: measured, reproduced, root causes identified in code. Found
2026-09-01 while benchmarking MTP; none of it is about MTP. Nothing has been
changed, unpublished, or announced.**

## RESOLVED (2026-09-01) — all three rungs work, at or near stock speed

**A machine-mixing error, caught before it shipped.** The first version of
this table said the rungs were 1.11-1.52x FASTER than stock. They are not:
the VQ figures were measured on the M3 Ultra and the stock baseline on the
M4 Max, and that baseline is 16.7 tok/s on the M4 against **23.7 on the M3**.
Comparing across machines invented a win that is not there. Everything below
is one machine, one prompt, one harness.

**M3 Ultra, `mlx-lm generate`, 128 greedy tokens, clean venv:**

| artifact | before | **after** | vs stock 23.7 | resident |
|---|---|---|---|---|
| `27B-VQ-3.9bpw` (d=4, packed 12) | 0.731 | **18.545** (25.4x) | 0.78x | 21.3 GB |
| `27B-VQ-4.5bpw` (d=2) | 7.818 | **24.227** (3.1x) | **1.02x** | **15.9 GB** |
| `27B-VQ-4.8bpw` (d=2, packed 9) | 1.070 | **25.322** (23.7x) | 1.07x | 30.0 GB |
| `Qwen3.8-27B-8bit` (stock) | — | 23.736 | 1.00x | 28.9 GB |

(As *shipped*, 4.5 and 4.8 crashed outright and 3.9 measured 0.426 tok/s on
the M4; the "before" column above is the M3, post-`model.py`-fix, so that the
kernel effect is isolated on one machine.)

The real result is 4.5bpw: **baseline speed at 55% of the memory**. 4.8bpw
edges ahead of stock. 3.9bpw is still 22% slower than stock, because d=4 must
keep its 64 KB codebook in device memory and pays per lookup for it.

Achieved bandwidth against the M3 Ultra's ~819 GB/s peak:

| artifact | GB/s | % peak |
|---|---|---|
| stock 8-bit | 641 | **78%** |
| 4.8bpw | 410 | 50% |
| 4.5bpw | 368 | 45% |
| 3.9bpw | 244 | 30% |

So the fallback is gone, but VQ still runs at 30-50% of peak where stock
reaches 78%. **There is roughly another 1.6-2.6x of headroom in the dense
path**, and the same is likely true of the MoE kernels.

Two kernels did it, both in `vq_switch.py`, both validated in
`tests/test_vq_dense_tiled.py`:

- **`_SRC_DENSE_D2_TILED` / `_SRC_DENSE_PACKED_D2_TILED`** — stage one block
  span of x instead of the whole row, so the threadgroup budget is
  `(K + 32*(G/2)) * 4` with no width term. Bit-identical to the untiled
  kernels wherever both are legal.
- **`_SRC_DENSE_D4_TILED` / `_SRC_DENSE_PACKED_D4_TILED`** — d=4 had no dense
  kernel at all, and at K=4096 its codebook is 64 KB, twice the entire
  threadgroup budget. These keep the codebook in device memory (L2-resident,
  the route `_SRC_FUSED_D8` already takes) and tile x. Bit-identical to the
  expert kernels they were derived from.

One gate bug was worth more than either kernel: `VQLinear` re-checked
`codebook.shape[1] == 2` alongside the fits test, so the d4 rung stayed on
the decode path even after its kernel existed. `dense_fits()` is now the
single authority on which geometries have a kernel.

**Still open:** the MoE VQ path sits at 15-16% of achieved bandwidth
(35B-A3B VQ 84 GB/s, 397B 86 GB/s) and its expert kernels use the same
whole-row staging. They fit today because expert widths are small, so nothing
falls back --- but the efficiency headroom the dense path just realised is
probably there too. That is the next piece of work, and it is also the
likeliest lever on the 397B's seq2/seq1 = 1.49 (docs/MTP.md).

The three causes below are what was wrong, kept for the record.

## What a downloader gets today

One machine (M4 Max / M3), one harness (`mlx_lm generate`), one prompt, 128
tokens, greedy, no drafting head anywhere.

| artifact | as shipped | with `model.py` regenerated | peak |
|---|---|---|---|
| `Qwen--Qwen3.8-27B-8bit` (stock reference) | **16.687 tok/s** | — | 28.9 GB |
| `27B-VQ-3.9bpw` | 0.426 tok/s | 0.731 tok/s | 21.5 GB |
| `27B-VQ-4.5bpw` | **crashes** | 7.818 tok/s | 15.9 GB |
| `27B-VQ-4.8bpw` | **crashes** | 1.070 tok/s | 30.0 GB |

Two of the three published dense rungs **cannot generate a single token** for
anyone who downloads them. The third runs at 1/39th of stock speed.

The MoE VQ path is fine, measured the same way — this is specific to dense:

| MoE artifact | tok/s |
|---|---|
| `Qwen3.6-35B-A3B-VQ-3.8bpw` | 72.1 |
| `Qwen3.8-Flash-Next-VQ-2.1bpw` | 19.5 |

## Cause 1 — the shipped `model.py` imports a module downloaders do not have

All three artifacts contain:

```python
from mlx_lm.models.vq_switch import _dense_fused
```

`mlx_lm.models.vq_switch` is ours; it does not exist in any released mlx-lm.
The self-contained `model.py` exists precisely so a downloader needs no local
code, and this line defeats that. It is reached the moment a layer takes the
fused path, i.e. on the first MLP forward — hence 4.5 and 4.8 crashing with
`ModuleNotFoundError` while 3.9 survives only because it never reaches the
fused branch at all (cause 2).

**This is already fixed in the repo.** `build_dense_vq.py` concatenates
`vq_switch.py + vq_dense.py + SHIM` into the bundle and `vq_dense._resolve_kernel`
prefers the local copy; the comment there describes this exact historical bug.
The artifacts on disk predate the fix.

**Fix: regenerate `model.py`. No re-fit, no re-quantisation, no new weights.**
Verified non-destructively — symlink the weights, drop in a freshly generated
`model.py`, and 4.5bpw goes from `ModuleNotFoundError` to 7.818 tok/s with
coherent output.

## Cause 2 — there is no dense fused kernel for `d=4`

`VQLinear.__call__` gates the fast path on `codebook.shape[1] == 2`. The
3.9bpw artifact is built at **d=4**, so *all 192* of its VQ linears take the
fallback, which decodes the entire weight matrix on every forward call.

That is the 39x. Regenerating `model.py` does not help it (0.426 -> 0.731
tok/s, same order), because the problem is not the import — it is that no
fused path exists for its geometry.

Either a d=4 dense kernel gets written, or dense artifacts stop shipping at
d=4.

## Cause 3 — wide MLP shapes exceed Metal's threadgroup cap even at `d=2`

The dense kernel caches both the codebook and the sub-vector array in
threadgroup memory, so it needs `(K + NSUB) * 4 + 1024` bytes against a 32768
cap. For the 27B MLP layers `IN = 17408`, so at d=2 `NSUB = 8704`:

| rung | shape | needs | verdict |
|---|---|---|---|
| 4.5 / 4.8 | 128x `IN=5120`, NSUB 2560 | 12–13 KB | fits, fused |
| 4.5 / 4.8 | 64x `IN=17408`, NSUB 8704 | **36–38 KB** | **too big, falls back** |

So even with cause 1 fixed, a third of the layers — and they are the widest
ones — still decode their whole weight per call. That is the residual 7.8
tok/s against stock's 16.7.

The budget is dominated by `NSUB`, which is a function of layer width. A
kernel that **tiles over NSUB** — blocking the sub-vector loop and
accumulating — removes that term from the threadgroup budget entirely and
would let every dense layer fuse at any width. That is the real fix and it
does not exist yet; `vq_dense.py` says as much in its own comments.

## Cause 3b — packed codes make the fallback dramatically worse

4.5bpw and 4.8bpw are structurally identical (d=2, 128 fused / 64 fallback)
and differ by one field: **4.8bpw sets `pack_bits=9`, 4.5bpw sets 0.** In the
fallback path packed codes are unpacked in full, per call, *before* the full
decode:

```python
if self.pack_bits:
    codes = _unpack_rows(codes, IN // self.codebook.shape[1], self.pack_bits)
w = _decode(codes, ...)
```

Two full materialisations of the weight per forward instead of one. That is
7.818 -> 1.070 tok/s and 15.9 -> 30.0 GB peak, from a single config flag.

Until a tiling kernel lands, `pack_bits` should not be set on layers that
cannot fuse.

## Why the smoke test passed

`vqlab smoke` is not the problem --- its docstring says "generate one token
through the runtime the artifact SHIPS", it loads the in-checkpoint
`model.py`, and one token is enough to reach the fused path. It would have
caught this.

It ran in an environment where the bad import resolves. Both build venvs on
the M4 carry a copy of our module inside the installed package:

    ~/vqvenv/lib/python3.12/site-packages/mlx_lm/models/vq_switch.py
    ~/qlab-venv/lib/python3.12/site-packages/mlx_lm/models/vq_switch.py

`~/.venvs/qwen4exp` (a clean install) does not, which is why the failure only
appeared now. So the smoke gave real assurance about a machine nobody
downloads to.

**Fixed 2026-09-01: `vqlab smoke --strict` (the default) now asserts.**
Printing the resolved runtime was never enough; the tool has to fail on it.
Strict rejects an artifact when:

- any VQ class resolves from outside the artifact directory;
- either fused kernel (`_dense_fused`, `_fused`) resolves from outside it ---
  checked separately, because the classes came from the artifact even in the
  broken builds and it was the KERNEL, resolved by name at call time, that
  came from elsewhere;
- `mlx_lm.models.vq_switch` exists at all. "Outside site-packages" is not the
  test and would have missed this: our build venvs carry that module INSIDE
  site-packages, and no downloader has it;
- mlx-lm is a repo checkout rather than an installed package.

Two implementation details that the first attempt got wrong, both now pinned
by `tests/test_smoke_strict.py`:

- mlx-lm loads a checkpoint's `model.py` via
  `spec_from_file_location("custom_model", ...)` and never registers it in
  `sys.modules`, so `sys.modules[cls.__module__]` is None for every bundled
  artifact --- the check silently skipped the exact case it exists for. The
  class's own functions close over the module dict, which does carry
  `__file__`.
- The "exec'd, no file" placeholder string must classify as *unknown*, not as
  a path; reading it as one failed a correct artifact.

Verified both ways against a real artifact: the shipped 3.9bpw rung reports
every class and kernel resolving from its own `model.py` and passes, while a
deliberately mis-built bundle carrying `vq_dense` only resolves its kernels
from the repo checkout and exits 1, naming the file.

## Bandwidth: it is not the bottleneck for VQ artifacts, and that is the point

Achieved memory bandwidth, taking active parameters x bytes-per-parameter as
the bytes a token must read, against the M4 Max's ~546 GB/s:

| model | GB/token | tok/s | GB/s | % of peak |
|---|---|---|---|---|
| 27B stock 8-bit (dense) | 27.00 | 16.69 | **450.6** | **83%** |
| 35B-A3B stock 8-bit (MoE) | 3.00 | 81.5 | 244.4 | 45% |
| 27B VQ 4.5bpw (model.py fixed) | 15.19 | 7.82 | 118.8 | 22% |
| 397B VQ 2.2bpw (MoE) | 4.67 | 18.3 | 85.7 | 16% |
| 35B-A3B VQ 3.8bpw (MoE) | 1.43 | 59.0 | 84.1 | 15% |
| 27B VQ 3.9bpw (every layer on fallback) | 13.16 | 0.43 | 5.7 | **1%** |

The stock dense model runs at 83% of peak --- it IS bandwidth-bound, exactly
as expected. **Every VQ artifact lands at 15-22%.** VQ moves the workload out
of the memory system and into the ALU: each weight costs a codebook lookup
and an fma instead of a shift-and-scale, and the current kernels are not
close to hiding that.

The sharpest single statement of the cost is the 35B pair, same model, both
resident with room to spare:

- stock 8-bit: 3.00 GB/token, **12.27 ms** per forward
- VQ 3.8bpw: 1.43 GB/token, **16.94 ms** per forward

**VQ reads 2.1x fewer bytes and runs 38% slower.** Per byte moved, the VQ
path is roughly 2.9x less efficient than stock's quantized matmul. The whole
promise of VQ at decode is fewer bytes per token, and today the kernels give
that win back and then some. That gap is the headroom, and it is large.

## The kernel work, in priority order

**P0 (no kernel work): regenerate `model.py`.** Two dead artifacts become
live. 4.5bpw measured at 7.82 tok/s this way.

**P1: tile the threadgroup staging of `x` over NSUB.** This is the single
highest-value kernel change and it fixes causes 2, 3 and 3b at once.

The d2 dense kernel stages the entire input vector in threadgroup memory:

```metal
threadgroup half2 cb[MAX_K];      // codebook: K * 4 bytes
threadgroup half2 xs[MAX_NSUB];   // the WHOLE input vector: NSUB * 4 bytes
```

so the allocation is `(K + NSUB) * 4 + 1024` and scales with **layer width**.
At `IN = 17408` the `xs` term alone is 34,816 bytes against a 32,768 cap.

Stage `x` in fixed tiles instead, looping over tiles and accumulating:

```metal
threadgroup half2 cb[MAX_K];
threadgroup half2 xs[TILE];       // fixed, e.g. 2048
for (int t0 = 0; t0 < NSUB; t0 += TILE) { load tile; barrier; accumulate; }
```

The budget becomes `(K + TILE) * 4 + 1024` --- **independent of layer width**.
At K=4096/TILE=2048 that is 25,600 bytes; at K=512, 11,264.

Two details that make it safe:

- **Align tiles to group boundaries.** Scales apply per group of G=64, i.e.
  32 sub-vectors, so a TILE that is a multiple of 32 keeps the per-group
  `acc = fma(srow[g], gacc, acc)` sequence in exactly its current order. The
  result stays **bit-exact**, which matters because every published score for
  these artifacts was produced by the existing path.
- **The same change unlocks d=4.** `vq_dense.py` notes the expert kernel
  (`_fused` with E=1) was tried as the d != 2 dense path and died at kernel
  load at 27B shapes needing 36,864 bytes --- the identical `xs[MAX_NSUB]`
  problem. Tiling it makes d=4 dense work, which is the only route to fixing
  3.9bpw without rebuilding its weights.

Expected gain is large but unmeasured: 4.5bpw already has 128 of 192 layers
fused and reaches 7.82 tok/s, so removing the remaining 64 whole-weight
decodes should move it substantially toward stock's 16.69. Do not quote a
number before measuring it.

**P2: close the 15-22% efficiency gap.** Even fully fused, VQ is ~2.9x less
efficient per byte than stock's quantized matmul. Directions worth profiling
before committing to any of them: wider codebook vectors so one lookup covers
more weights (needs dense kernels for d>2), keeping the codebook in registers
or simdgroup storage rather than threadgroup, and using simdgroup matrix
operations instead of scalar per-row accumulation. This is a research task,
not a patch, and P0/P1 should land first.

## Suggested order of work

0. **Move the release smoke to a clean-mlx-lm venv.** Without this, the same
   class of defect ships again.
1. **Regenerate `model.py` for all three rungs and republish.** Cheapest
   possible fix, turns two dead artifacts into working ones, no weights
   change. (4.5bpw -> 7.8 tok/s, 4.8bpw -> 1.1 tok/s.)
2. **Do not ship dense at d=4** until a d=4 kernel exists, or rebuild 3.9bpw
   at d=2. It is the slowest of the three and the least fixable as built.
3. **Drop `pack_bits` on non-fusing layers** — a config change worth ~7x on
   4.8bpw.
4. **Write the NSUB-tiled dense kernel.** This is the one that actually
   closes the gap to stock, and it also removes the shape ceiling that the
   guard currently exists to detect.
5. Add a release gate that generates a token from the artifact **through a
   clean mlx-lm**, not a dev tree. Every failure above would have been caught
   by one greedy token from a venv without our patched mlx-lm.

## Caveats

The per-rung timings are single runs on machines that were also doing other
work, so treat them as order-of-magnitude, not precision figures. The
qualitative results — crash vs no crash, 39x, the 7x pack_bits gap, the
memory doubling — are far outside any contention noise, and the mechanisms
are identified in the source rather than inferred from the timings.
