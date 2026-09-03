#!/usr/bin/env python
"""Measure the per-DISPATCH launch floor of the VQ decode path, and test the
two structural dispatch-reduction candidates.  (2026-09-02, dispatch arc.)

Why this exists.  Two fingerprints suggested VQ decode might be bound by
kernel-launch latency rather than bytes or FLOPs: the packed-d8 kernel sits
at 61-97 GB/s of an 819 GB/s machine with no explanation for the absolute
floor, and on the cluster VQ 2.6bpw and 3.1bpw decode at the same speed
despite a 21 GiB size difference.  A decode step dispatches ~144 VQ kernels
(48 layers x gate/up/down).  If a launch costs the ~14 us an earlier arc
reported, that is 2 ms/token and worth chasing.

It is not.  Every arm below refutes the hypothesis; the script is kept so the
negative is reproducible without re-deriving it.  See the ledger entry
"2026-09-02 -- DISPATCH-REDUCTION ARC" in
research/quantlab/research/flash-next/LEDGER.md.

METHODOLOGY (the arc standard, and two traps it avoids).
  * every variant is JIT-warmed before timing;
  * variants are timed INTERLEAVED (A,B,A,B) inside each rep so a contention
    burst hits both arms;
  * headline is MIN over reps with the MEDIAN alongside; a claimed win must
    hold on both;
  * mx.eval is called on the FULL list of outputs -- evaluating only the last
    array leaves the rest dead and measures submit overhead (~14 us/shape,
    2.5 TB/s: the number that motivated this arc);
  * mx.eval is NEVER called per dispatch -- that measures ~380 us of sync.

A THIRD TRAP, found here: an INDEPENDENT chain of M real dispatches holds M
live output arrays, where the dependent chain of the same M holds one.  At
M=144 on the [10,640] shapes that asymmetry is itself measurable -- the
independent arm inflates from ~34 to ~62-80 us/dispatch and the dep/indep
ratio inverts below 1.  M=48 is the honest comparison point for arms C and B;
the M=144 independent rows are printed but are allocation-contaminated and
must not be read as a launch measurement.  The trivial-kernel arm A is
unaffected (its outputs are tiny), which is why the floor is quoted from A.

Run:  PYTHONPATH=src python scripts/bench_dispatch_floor.py
Needs the 2.1bpw artifact readable; loads three tensors (E sliced to 64,
~1 GiB), never a model.
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from vqlab import vq_switch as V

ART_DEFAULT = ("/Volumes/Thunderbay SSD/Exo Models/"
               "TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw")
LAYER = 20        # a d8/K16384 14-bit packed layer: the hot geometry (96%)
E_SLICE = 64      # expert-axis slice; keeps resident bytes ~1 GiB
LAYERS = 46       # d8 layers in the 2.1bpw
TOPK = 10


def load_proj(art, layer, proj, e=E_SLICE):
    idx = json.load(open(os.path.join(art, "model.safetensors.index.json")))
    idx = idx["weight_map"]
    out = {}
    for part in ("codes", "codebook", "vq_scales"):
        k = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{part}"
        w = mx.load(os.path.join(art, idx[k]))[k]
        if part != "codebook" and e is not None:
            w = w[:e]
        out[part] = mx.contiguous(w)
    mx.eval(list(out.values()))
    return out


def interleaved(arms, reps=15, warm=2):
    """arms: {name: thunk building one lazy unit}. Returns {name: (min, med)}."""
    for fn in arms.values():
        for _ in range(warm):
            mx.eval(fn())
    acc = {n: [] for n in arms}
    for _ in range(reps):
        for name, fn in arms.items():
            o = fn()
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(o)
            acc[name].append(time.perf_counter() - t0)
    return {n: (min(v), statistics.median(v)) for n, v in acc.items()}


_NULL = mx.fast.metal_kernel(
    name="vq_bench_null", input_names=["inp"], output_names=["out"],
    source="out[thread_position_in_grid.x] = inp[thread_position_in_grid.x]"
           " + T(1);",
    header="", ensure_row_contiguous=True)


def _null(a, n):
    (y,) = _NULL(inputs=[a], template=[("T", mx.float16)], grid=(n, 1, 1),
                 threadgroup=(min(n, 256), 1, 1),
                 output_shapes=[(n,)], output_dtypes=[mx.float16])
    return y


def bench_null_floor(reps):
    """A. The launch floor itself, on a kernel that does one store.

    Both a DEPENDENT chain (what decode is) and an INDEPENDENT one (which the
    GPU may overlap).  The dependent number is the honest floor.
    """
    print("\n=== A. launch floor: trivial kernel, us/dispatch (min/median) ===")
    print("     grid      M   dependent   independent")
    a = mx.zeros((65536,), dtype=mx.float16)
    mx.eval(a)
    for n in (1, 1024, 65536):
        for M in (48, 144, 288):
            def dep(M=M, n=n):
                y = a[:n]
                for _ in range(M):
                    y = _null(y, n)
                return y

            def ind(M=M, n=n):
                return [_null(a[:n], n) for _ in range(M)]

            r = interleaved({"dep": dep, "ind": ind}, reps=reps)
            d, i = r["dep"], r["ind"]
            print(f"   {n:6d} {M:5d}   {d[0]/M*1e6:5.2f}/{d[1]/M*1e6:5.2f}"
                  f"   {i[0]/M*1e6:5.2f}/{i[1]/M*1e6:5.2f}")


def bench_real_floor(t, reps):
    """B. The same question on the REAL packed-d8 kernel.

    M is swept so the per-dispatch cost separates from the eval round-trip,
    and OUT is swept so the kernel's own FIXED cost (its intercept) separates
    from the per-row work.
    """
    codes, cb, sc = t["codes"], t["codebook"], t["vq_scales"]
    E, OUT, _ = codes.shape
    IN = sc.shape[2] * 64
    x = mx.random.normal((TOPK, IN)).astype(mx.float16)
    eidx = mx.array([i % E for i in range(TOPK)], dtype=mx.uint32)
    mx.eval(x, eidx)

    def one(c=codes, s=sc, xx=None):
        return V._fused(xx if xx is not None else x, eidx, c, cb, s,
                        pack_bits=14)

    print(f"\n=== B. real gate_proj (OUT={OUT}, IN={IN}), N={TOPK} ===")
    print("   M sweep, independent chain, us/dispatch (min/median)")
    for M in (1, 8, 48, 144):
        r = interleaved({"f": lambda M=M: [one() for _ in range(M)]}, reps=reps)
        mn, md = r["f"]
        print(f"     M={M:4d}   {mn/M*1e6:7.2f} / {md/M*1e6:7.2f}")

    print("   OUT sweep at M=48 -- the intercept is the kernel's fixed cost")
    for rows in (8, 32, 128, 512, OUT):
        c2 = mx.contiguous(codes[:, :rows, :])
        s2 = mx.contiguous(sc[:, :rows, :])
        mx.eval(c2, s2)
        r = interleaved({"f": lambda: [one(c2, s2) for _ in range(48)]},
                        reps=reps)
        mn, md = r["f"]
        print(f"     OUT={rows:5d}   {mn/48*1e6:7.2f} / {md/48*1e6:7.2f}")
        del c2, s2


def bench_dependence(g, d, reps):
    """C. What dependence costs on the real kernel.

    gate (2560->640) and down (640->2560) ping-pong with ZERO glue ops, so the
    dependent arm is pure kernel-to-kernel serialisation against an
    independent arm running the identical kernel mix.
    """
    E = g["codes"].shape[0]
    IN = 2560
    x = mx.random.normal((TOPK, IN)).astype(mx.float16)
    x640 = mx.zeros((TOPK, 640), dtype=mx.float16)
    eidx = mx.array([i % E for i in range(TOPK)], dtype=mx.uint32)
    mx.eval(x, x640, eidx)

    def f(t, xx):
        return V._fused(xx, eidx, t["codes"], t["codebook"], t["vq_scales"],
                        pack_bits=14)

    print("\n=== C. dependent vs independent, real kernels, N=%d ===" % TOPK)
    print("   (read M=48; the M=144 independent arm holds 144 live outputs "
          "and is\n    allocation-contaminated -- see the module docstring)")
    print("      M   dependent   independent   ratio")
    for M in (48, 144):
        def dep(M=M):
            y = x
            for i in range(M):
                y = f(g if i % 2 == 0 else d, y)
            return y

        def ind(M=M):
            return [f(g, x) if i % 2 == 0 else f(d, x640) for i in range(M)]

        r = interleaved({"dep": dep, "ind": ind}, reps=reps)
        (a0, a1), (b0, b1) = r["dep"], r["ind"]
        print(f"   {M:4d}   {a0/M*1e6:6.2f}/{a1/M*1e6:6.2f}"
              f"   {b0/M*1e6:6.2f}/{b1/M*1e6:6.2f}   {a0/b0:5.2f}x")


def bench_fusion(g, u, d, reps):
    """D. CANDIDATE (a): fuse gate_proj+up_proj into one doubled-OUT dispatch.

    Chain of 46 real layer-shaped MLP steps, dependent layer to layer.  The
    fused arm concatenates codes/scales along OUT -- exactly the geometry a
    real fusion would build -- but reuses gate's codebook for both halves, so
    it is charged NOTHING for the two-codebook selection a real fusion needs.
    It is therefore an UPPER BOUND on the prize.  Cost only; no bit-identity
    is claimed for this arm and none is needed to read the result.
    """
    E, HID = g["codes"].shape[0], 640
    IN = 2560
    gu_c = mx.contiguous(mx.concatenate([g["codes"], u["codes"]], axis=1))
    gu_s = mx.contiguous(mx.concatenate([g["vq_scales"], u["vq_scales"]],
                                        axis=1))
    mx.eval(gu_c, gu_s)

    print("\n=== D. candidate (a): gate+up fusion, UPPER BOUND ===")
    print(f"   {LAYERS} layer-steps, fused OUT {g['codes'].shape[1]} ->"
          f" {gu_c.shape[1]}")
    print("       N    unfused ms    fused ms    saving")
    for N in (1, 5, 10, 20):
        x = mx.random.normal((N, IN)).astype(mx.float16)
        eidx = mx.array([i % E for i in range(N)], dtype=mx.uint32)
        mx.eval(x, eidx)

        def f(c, cb, s, xx):
            return V._fused(xx, eidx, c, cb, s, pack_bits=14)

        def unfused():
            y = x
            for _ in range(LAYERS):
                a = f(g["codes"], g["codebook"], g["vq_scales"], y)
                b = f(u["codes"], u["codebook"], u["vq_scales"], y)
                h = nn.silu(a) * b
                y = f(d["codes"], d["codebook"], d["vq_scales"], h)
            return y

        def fused():
            y = x
            for _ in range(LAYERS):
                ab = f(gu_c, g["codebook"], gu_s, y)
                h = nn.silu(ab[:, :HID]) * ab[:, HID:]
                y = f(d["codes"], d["codebook"], d["vq_scales"], h)
            return y

        r = interleaved({"unfused": unfused, "fused": fused}, reps=reps)
        (u0, u1), (f0, f1) = r["unfused"], r["fused"]
        print(f"    {N:4d}   {u0*1e3:7.3f}({u1*1e3:7.3f})"
              f"  {f0*1e3:7.3f}({f1*1e3:7.3f})"
              f"  {(1-f0/u0)*100:6.1f}% [med {(1-f1/u1)*100:5.1f}%]")


def bench_compile(g, u, d, reps):
    """E. CANDIDATE (b): does mx.compile coalesce the per-layer VQ sequence?

    Three arms over the SAME work: raw _fused calls, the real
    VQSwitchLinear.__call__ (broadcast/reshape/astype glue included), and that
    module step under mx.compile.
    """
    from vqlab.vq_switch import VQSwitchLinear
    IN = 2560
    mods = {n: VQSwitchLinear.from_weights(t["codes"], t["codebook"],
                                           t["vq_scales"])
            for n, t in (("g", g), ("u", u), ("d", d))}
    x1 = mx.random.normal((1, 1, IN)).astype(mx.float16)
    indices = mx.array([[list(range(TOPK))]], dtype=mx.uint32)
    eidx = mx.array(list(range(TOPK)), dtype=mx.uint32)
    xflat = mx.broadcast_to(x1.reshape(1, IN), (TOPK, IN))
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

    def step(y):
        h = nn.silu(mods["g"](y, indices)) * mods["u"](y, indices)
        return mods["d"](h, indices)

    cstep = mx.compile(step)

    def chain(s):
        def go():
            y = x1
            for _ in range(LAYERS):
                y = s(y).sum(axis=2).reshape(1, 1, IN)
            return y
        return go

    r = interleaved({"raw": raw, "module": chain(step),
                     "compiled": chain(cstep)}, reps=reps)
    print("\n=== E. candidate (b): mx.compile over the layer step ===")
    print(f"   {LAYERS} layer-steps, top_k={TOPK} (one token's expert path)")
    base = r["raw"][0]
    for n in ("raw", "module", "compiled"):
        mn, md = r[n]
        print(f"     {n:9s} {mn*1e3:7.3f} ms (med {md*1e3:7.3f})"
              f"   {mn-base and (mn-base)*1e3:+7.3f} ms vs raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=ART_DEFAULT)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--only", default="", help="comma list of A,B,C,D,E")
    a = ap.parse_args()
    want = set(a.only.upper().split(",")) if a.only else set("ABCDE")

    print(f"artifact: {a.artifact}\nlayer {LAYER}, E sliced to {E_SLICE}, "
          f"reps={a.reps}")
    if "A" in want:
        bench_null_floor(a.reps)
    if want & set("BCDE"):
        g = load_proj(a.artifact, LAYER, "gate_proj")
        u = load_proj(a.artifact, LAYER, "up_proj")
        d = load_proj(a.artifact, LAYER, "down_proj")
        if "B" in want:
            bench_real_floor(g, a.reps)
        if "C" in want:
            bench_dependence(g, d, a.reps)
        if "D" in want:
            bench_fusion(g, u, d, a.reps)
        if "E" in want:
            bench_compile(g, u, d, a.reps)


if __name__ == "__main__":
    main()
