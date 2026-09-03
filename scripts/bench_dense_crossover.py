#!/usr/bin/env python
"""Where does the DENSE fused kernel stop beating decode-then-GEMM?
(2026-09-03, dense-kernel arc.)

_DENSE_FUSED_MAX_N_PACKED defaults to 96, and its measurement note in
vq_dense.py is explicit that it was taken on the 27B **d=2** mlp shape
(K512/bits9). The 3.9bpw rung is d=4/K=4096/packed-12 and got a dense kernel
later, so its own crossover was never measured -- and BOTH sides of the
comparison have since moved: the fused side by 1.6-1.7x (devx), the decode
side by the row-tiled bounded-transient path.

So this re-measures the crossover for each rung on disk, with both current
paths, one real VQLinear at a time. Two answers per rung: where fused stops
winning, and what the cutoff costs if left where it is.

Run:  PYTHONPATH=src python scripts/bench_dense_crossover.py
"""
import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx

from vqlab import vq_dense as VD

MODELS = os.path.expanduser("~/.exo/models")
RUNGS = (
    ("3.9bpw", "TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw"),
    ("4.5bpw", "TheDrainFlorist--Qwen3.8-27B-VQ-4.5bpw"),
    ("4.8bpw", "TheDrainFlorist--Qwen3.8-27B-VQ-4.8bpw"),
)


def load(art, layer, proj):
    idx = json.load(open(os.path.join(
        art, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for part in ("codes", "codebook", "vq_scales"):
        k = f"language_model.model.layers.{layer}.mlp.{proj}.{part}"
        out[part] = mx.contiguous(mx.load(os.path.join(art, idx[k]))[k])
    mx.eval(list(out.values()))
    return out


def timeit(fn, reps=9, warm=2):
    for _ in range(warm):
        mx.eval(fn())
    v = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        v.append(time.perf_counter() - t0)
    return min(v) * 1e3, statistics.median(v) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="gate_proj")
    ap.add_argument("--ns", default="1,16,32,64,96,128,192,256,512")
    a = ap.parse_args()
    ns = [int(s) for s in a.ns.split(",")]
    for tag, sub in RUNGS:
        art = os.path.join(MODELS, sub)
        if not os.path.isdir(art):
            continue
        cfg = json.load(open(os.path.join(art, "config.json")))
        geom = next(iter((cfg.get("vq_linear") or {}).values()), {})
        bits, group = geom.get("pack_bits") or 0, geom.get("group", 64)
        t = load(art, 20, a.proj)
        IN = t["vq_scales"].shape[1] * group
        m = VD.VQLinear(t["codes"], t["codebook"], t["vq_scales"],
                        group_size=group, pack_bits=bits,
                        in_features=IN if bits else None)
        cut = (VD._DENSE_FUSED_MAX_N_PACKED if bits
               else VD._DENSE_FUSED_MAX_N_PLAIN)
        print(f"\n### {tag}  d={geom.get('dim')} K={geom.get('k')} "
              f"pack_bits={bits}  {a.proj} OUT={m.output_dims} IN={IN}"
              f"   shipped cutoff N<={cut}")
        print(f"{'N':>5s} {'fused_ms':>18s} {'decode_ms':>18s} "
              f"{'fused/decode':>13s} {'ident':>6s}")
        for N in ns:
            mx.random.seed(N)
            x = mx.random.normal((N, IN)).astype(mx.float16)
            mx.eval(x)
            try:
                VD._DENSE_FUSED_MAX_N_PACKED = 10 ** 9
                VD._DENSE_FUSED_MAX_N_PLAIN = 10 ** 9
                yf = m(x)
                mx.eval(yf)
                f = timeit(lambda: m(x))
            except Exception as e:                       # kernel refused it
                print(f"{N:5d}  fused unavailable: {type(e).__name__}: {e}")
                f, yf = None, None
            VD._DENSE_FUSED_MAX_N_PACKED = 0
            VD._DENSE_FUSED_MAX_N_PLAIN = 0
            yd = m(x)
            mx.eval(yd)
            d = timeit(lambda: m(x))
            if f is None:
                continue
            ident = "YES" if bool(mx.array_equal(
                yf.view(mx.uint16), yd.view(mx.uint16))) else "no"
            print(f"{N:5d} {f[0]:8.3f}/{f[1]:8.3f} {d[0]:8.3f}/{d[1]:8.3f} "
                  f"{f[0]/d[0]:12.3f} {ident:>6s}")
            del yf, yd
            mx.clear_cache()
        VD._DENSE_FUSED_MAX_N_PACKED = 96
        VD._DENSE_FUSED_MAX_N_PLAIN = 12
        del t, m
        mx.clear_cache()
    print("\nfused/decode < 1 means the FUSED kernel is still winning.")


if __name__ == "__main__":
    if not os.path.isdir(MODELS):
        sys.exit("no local models dir")
    main()
