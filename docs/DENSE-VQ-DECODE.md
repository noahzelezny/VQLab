# The dense 27B VQ artifacts are broken and slow — three distinct causes

**Status: measured, reproduced, root causes identified in code. Found
2026-09-01 while benchmarking MTP; none of it is about MTP. Nothing has been
changed, unpublished, or announced.**

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

## Suggested order of work

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
