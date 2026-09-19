#!/usr/bin/env python
"""Micro-bench ONE GatedResidual at decode shape: is mx.compile inert here?

WARNING, F133: the CHAIN-SPLIT section of this file is INVALID as written.
Its parts sum to 1667 us against a 410 us whole, and a projection to four
outputs measured slower than the entire chain -- it times per-mx.eval
round-trip overhead, not the ops. Only the PAIRED compile ratio survives,
because that overhead cancels across both arms. Fix before trusting the
split: amortize many ops per eval instead of one eval per op.

F133's whole-model hc-compile arm measured +0.66% (a null), but a
numerics-preserving arm's checksum is identical BY DESIGN, so it could not
distinguish "fusion does not help" from "mx.compile silently no-op'd".
This isolates the question: one module, 10k iterations, compiled vs not.
No artifact, no 47 GB load, seconds to run.

Also splits the chain, so the 22.87 ms F131 attributed to hyper-connections
gets an internal breakdown instead of a guess.
"""
import time
import mlx.core as mx
import mlx.nn as nn

HC, D, LOWRANK = 4, 2560, 320
HCDIM = HC * D
N = 2000


def bench(label, fn, x, n=N):
    for _ in range(50):                      # warm + (for compile) trace
        mx.eval(fn(x))
    mx.synchronize()
    t0 = time.time()
    for _ in range(n):
        mx.eval(fn(x))
    mx.synchronize()
    us = (time.time() - t0) / n * 1e6
    print(f"  {label:<34} {us:8.1f} us/call")
    return us


class HC_(nn.Module):
    """The mlx_lm GatedResidual chain, 8-bit affine like the artifact."""
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(HCDIM)
        self.down = nn.Linear(HCDIM, LOWRANK, bias=False)
        self.up = nn.Linear(LOWRANK, HCDIM, bias=False)
        self.inject = nn.Linear(HCDIM, HC, bias=False)

    def __call__(self, h):
        normed = self.norm(h)
        w = nn.silu(self.down(normed) / HC)
        w = mx.sigmoid(self.up(w))
        w = w.reshape(*w.shape[:-1], HC, D)
        mixed = (w * normed.reshape(*normed.shape[:-1], HC, D)).mean(axis=-2)
        inj = 2 * mx.sigmoid(self.inject(normed) / HC)
        return mixed, h, inj


x = mx.random.normal((1, 1, HCDIM)).astype(mx.float16)

for bits in (None, 8):
    m = HC_()
    tag = "bf16/fp16" if bits is None else f"affine {bits}-bit"
    if bits:
        nn.quantize(m, group_size=64, bits=bits)
    mx.eval(m.parameters())
    print(f"\n=== full chain, {tag} ===")
    plain = bench("plain", lambda h, _m=m: _m(h)[0], x)
    comp_f = mx.compile(lambda h, _m=m: _m(h)[0])
    comp = bench("mx.compile", comp_f, x)
    print(f"  -> compile ratio {plain/comp:5.3f}x "
          f"({'HELPS' if plain/comp > 1.05 else 'INERT'})")

# Which part of the chain costs? Same shapes, 8-bit, decode batch.
m = HC_(); nn.quantize(m, group_size=64, bits=8); mx.eval(m.parameters())
print("\n=== chain split (affine 8-bit) ===")
bench("rmsnorm only", lambda h: m.norm(h), x)
bench("down 10240->320", lambda h: m.down(h), x)
bench("up 320->10240", lambda h: m.up(mx.zeros((1, 1, LOWRANK), mx.float16)), x)
bench("inject 10240->4", lambda h: m.inject(h), x)
bench("elementwise tail only",
      lambda h: (h.reshape(1, 1, HC, D) * h.reshape(1, 1, HC, D)).mean(axis=-2), x)
