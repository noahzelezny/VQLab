#!/usr/bin/env python
"""M1d unit test: VQSwitchLinear must be call-compatible with SwitchLinear.

Builds a small synthetic VQ format + the equivalent dense weights, then
checks outputs match through BOTH SwitchGLU regimes:
  - decode (indices.size < 64, unsorted, broadcast x)
  - prefill (indices.size >= 64 -> mlx_lm sorts, sorted_indices=True)
and through the loader hook (tree_unflatten path swap on a toy module).
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.vq_switch import VQSwitchLinear
from mlx_lm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort

rng = np.random.default_rng(7)
E, OUT, IN, D, K, G = 32, 192, 256, 4, 256, 64

codes = rng.integers(0, K, (E, OUT, IN // D)).astype(np.uint8)
cb = (rng.standard_normal((K, D)) * 0.05).astype(np.float16)
sc = (np.abs(rng.standard_normal((E, OUT, IN // G))) * 0.4 + 1e-3).astype(np.float16)

# dense equivalent
W = (cb.astype(np.float32)[codes].reshape(E, OUT, IN // G, G)
     * sc.astype(np.float32)[:, :, :, None]).reshape(E, OUT, IN)

vq = VQSwitchLinear(mx.array(codes), mx.array(cb), mx.array(sc))
dense = SwitchLinear(IN, OUT, E, bias=False)
dense.weight = mx.array(W.astype(np.float16))

ok = True


def close(tag, a, b, tol):
    global ok
    d = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
    m = float(mx.abs(b.astype(mx.float32)).max())
    rel = d / max(m, 1e-9)
    good = rel < tol
    ok &= good
    print(f"  {tag:38s} shape {tuple(a.shape)}  max rel {rel:.2e} "
          f"({'OK' if good else 'FAIL'})")


# regime 1: decode — x [B,T,1,1,IN], indices [B,T,k], unsorted
x = mx.array(rng.standard_normal((2, 3, 1, 1, IN)).astype(np.float16))
idx = mx.array(rng.integers(0, E, (2, 3, 4)).astype(np.uint32))
close("decode: broadcast unsorted", vq(x, idx), dense(x, idx), 2e-3)

# regime 2: prefill — sorted flat pairs like SwitchGLU's do_sort branch
T, k = 700, 8
xb = mx.array(rng.standard_normal((1, T, 1, 1, IN)).astype(np.float16))
idxb = mx.array(rng.integers(0, E, (1, T, k)).astype(np.uint32))
xs, idxs, inv = _gather_sort(xb, idxb)
y_vq = _scatter_unsort(vq(xs, idxs, sorted_indices=True), inv, idxb.shape)
y_dn = _scatter_unsort(dense(xs, idxs, sorted_indices=True), inv, idxb.shape)
close("prefill: sorted (5600 pairs)", y_vq, y_dn, 2e-3)

# regime 3: mid-size unsorted large-N path (internal sort branch)
xm = mx.array(rng.standard_normal((1, 900, 1, 1, IN)).astype(np.float16))
idxm = mx.array(rng.integers(0, E, (1, 900, 8)).astype(np.uint32))
close("large-N unsorted (internal sort)", vq(xm, idxm), dense(xm, idxm), 2e-3)

# loader hook path swap on a toy container
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = SwitchLinear(IN, OUT, E, bias=False)

from mlx.utils import tree_unflatten
toy = Toy()
toy.update_modules(tree_unflatten([
    ("proj", VQSwitchLinear(mx.array(codes), mx.array(cb), mx.array(sc)))]))
assert isinstance(toy.proj, VQSwitchLinear), "module swap failed"
close("after update_modules swap", toy.proj(x, idx), dense(x, idx), 2e-3)

print("\nM1d module test:", "ALL OK" if ok else "FAILURES ABOVE")
raise SystemExit(0 if ok else 1)
