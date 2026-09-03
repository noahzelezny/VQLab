#!/usr/bin/env python
"""Where VQSwitchLinear.__call__'s 1.07 ms/token of module glue goes.
(2026-09-02, kernel arc 5.)

Arc 3 SS4 measured the decode expert path at 5.84 ms raw `_fused` calls vs
6.90 ms through the real modules -- +1.07 ms/token of Python/graph glue on 144
module calls, and mx.compile cannot touch it (metal_kernel dispatches are
opaque).  Nobody attributed it.  This script does, then re-measures after the
fast path.

METHOD.  The glue is HOST-side: Python execution plus lazy-graph node
construction.  So every arm is timed in TWO phases per rep:

    build   mx.synchronize(); t0; construct the whole 46-layer lazy chain; t1
    eval    mx.eval(out); t2

The GPU cannot start before the first eval touches the graph, so `build` is a
clean host-glue number, and (t2-t0) is the end-to-end the decode loop pays.
Arms are interleaved A,B,A,... per rep; headline is min over reps with median
alongside (arc standard).

ARMS
    raw      _fused with prebuilt flat x/eidx    (the floor arc 3 quoted)
    module   the real VQSwitchLinear.__call__ chain
    micro    per-op attribution of one __call__'s glue pieces (host-only
             timing of each lazy op, x1000)

Run:  PYTHONPATH=src python scripts/bench_module_glue.py
Reads L20 gate/up/down of the 2.1bpw READ-ONLY (E sliced to 64); no model.
"""
import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from vqlab import vq_switch as V
from vqlab.vq_switch import VQSwitchLinear

from bench_dispatch_floor import ART_DEFAULT, LAYER, LAYERS, TOPK, load_proj


def two_phase(arms, reps=21, warm=2):
    for fn in arms.values():
        for _ in range(warm):
            mx.eval(fn())
    acc = {n: ([], []) for n in arms}
    for _ in range(reps):
        for name, fn in arms.items():
            mx.synchronize()
            t0 = time.perf_counter()
            o = fn()
            t1 = time.perf_counter()
            mx.eval(o)
            t2 = time.perf_counter()
            acc[name][0].append(t1 - t0)
            acc[name][1].append(t2 - t0)
    return {n: ((min(b), statistics.median(b)), (min(t), statistics.median(t)))
            for n, (b, t) in acc.items()}


def build_mods(g, u, d):
    return {n: VQSwitchLinear.from_weights(t["codes"], t["codebook"],
                                           t["vq_scales"])
            for n, t in (("g", g), ("u", u), ("d", d))}


def bench_chain(g, u, d, mods, reps):
    IN = 2560
    x1 = mx.random.normal((1, 1, IN)).astype(mx.float16)
    indices = mx.array([[list(range(TOPK))]], dtype=mx.uint32)
    eidx = mx.array(list(range(TOPK)), dtype=mx.uint32)
    xflat = mx.contiguous(mx.broadcast_to(x1.reshape(1, IN), (TOPK, IN)))
    mx.eval(x1, indices, eidx, xflat)

    def raw():
        y = xflat
        for _ in range(LAYERS):
            a = V._fused(y, eidx, g["codes"], g["codebook"], g["vq_scales"],
                         pack_bits=14)
            b = V._fused(y, eidx, u["codes"], u["codebook"], u["vq_scales"],
                         pack_bits=14)
            h = nn.silu(a) * b
            y = V._fused(h, eidx, d["codes"], d["codebook"], d["vq_scales"],
                         pack_bits=14)
        return y

    def module():
        y = x1
        for _ in range(LAYERS):
            h = nn.silu(mods["g"](y, indices)) * mods["u"](y, indices)
            y = mods["d"](h, indices).sum(axis=2).reshape(1, 1, IN)
        return y

    r = two_phase({"raw": raw, "module": module}, reps=reps)
    print(f"\n=== {LAYERS} layer-steps, top_k={TOPK}: build (host glue) vs "
          "total, ms min(med) ===")
    for n, ((b0, b1), (t0_, t1_)) in r.items():
        print(f"   {n:8s} build {b0*1e3:6.3f}({b1*1e3:6.3f})   "
              f"total {t0_*1e3:6.3f}({t1_*1e3:6.3f})")
    return r


def bench_micro(g, mods, reps=7, M=1000):
    """Host cost of each glue op in one __call__, x M, us/op (min over reps).
    All lazy -- nothing evaluated; this is graph-build + Python cost only."""
    IN = 2560
    x1 = mx.random.normal((1, 1, IN)).astype(mx.float16)
    indices = mx.array([[list(range(TOPK))]], dtype=mx.uint32)
    mx.eval(x1, indices)
    m = mods["g"]
    idx_flat = indices.flatten()
    eidx = idx_flat.astype(mx.uint32)
    xf = mx.broadcast_to(x1, (*indices.shape, 1, IN)).reshape(TOPK, IN)
    y = V._fused(xf, eidx, g["codes"], g["codebook"], g["vq_scales"],
                 pack_bits=14)
    mx.eval(y)

    ops = {
        "flatten": lambda: indices.flatten(),
        "astype_u32(u32)": lambda: idx_flat.astype(mx.uint32),
        "broadcast+reshape": lambda: mx.broadcast_to(
            x1, (*indices.shape, 1, IN)).reshape(TOPK, IN),
        "dims_mx.array": lambda: mx.array([640, 2560, 8, 64, TOPK, 16384],
                                          dtype=mx.int32),
        "module_getitem_x3": lambda: (m["codes"], m["codebook"],
                                      m["vq_scales"]),
        "input_dims_prop": lambda: m.input_dims,
        "out_astype+reshape": lambda: y.astype(mx.float16).reshape(
            *indices.shape, 1, 640),
        "kernel_call(_fused)": lambda: V._fused(
            xf, eidx, g["codes"], g["codebook"], g["vq_scales"],
            pack_bits=14),
        "full__call__": lambda: m(x1, indices),
    }
    print(f"\n=== host-side us/op (lazy, never evaluated), x{M}, "
          f"min(med) over {reps} ===")
    for n, fn in ops.items():
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(M):
                fn()
            ts.append((time.perf_counter() - t0) / M * 1e6)
        print(f"   {n:20s} {min(ts):7.2f} ({statistics.median(ts):7.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=ART_DEFAULT)
    ap.add_argument("--reps", type=int, default=21)
    a = ap.parse_args()
    print(f"artifact: {a.artifact}  layer {LAYER}")
    g = load_proj(a.artifact, LAYER, "gate_proj")
    u = load_proj(a.artifact, LAYER, "up_proj")
    d = load_proj(a.artifact, LAYER, "down_proj")
    mods = build_mods(g, u, d)
    bench_chain(g, u, d, mods, a.reps)
    bench_micro(g, mods)


if __name__ == "__main__":
    main()
