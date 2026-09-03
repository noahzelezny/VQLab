#!/usr/bin/env python
"""Acceptance gate for the dense devx kernels (2026-09-03, dense-kernel arc).

Asserts, through the REAL dispatcher (V._dense_fused, not a bench replica),
that the new default dense path is BIT-IDENTICAL to the pre-arc staged path
on REAL tensors from every dense rung on disk -- then times the pair.

Covers all three 27B rungs, i.e. four of the six dense kernels:
    3.9bpw  d4/K4096/packed-12   -> packed_d4_tiled
    4.5bpw  d2/K256/unpacked     -> d2 (gate/up, untiled) + d2_tiled (down)
    4.8bpw  d2/K512/packed-9     -> packed_d2 + packed_d2_tiled

The two kernels no artifact uses (unpacked d4) are covered synthetically in
tests/test_vq_dense_devx.py.

Run:  PYTHONPATH=src python scripts/dense_devx_accept.py
"""
import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx

from vqlab import vq_switch as V

MODELS = os.path.expanduser("~/.exo/models")
RUNGS = (
    ("3.9bpw", "TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw", 12),
    ("4.5bpw", "TheDrainFlorist--Qwen3.8-27B-VQ-4.5bpw", 0),
    ("4.8bpw", "TheDrainFlorist--Qwen3.8-27B-VQ-4.8bpw", 9),
)
PROJS = ("gate_proj", "up_proj", "down_proj")


def load(art, layer, proj):
    idx = json.load(open(os.path.join(
        art, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for part in ("codes", "codebook", "vq_scales"):
        k = f"language_model.model.layers.{layer}.mlp.{proj}.{part}"
        if k not in idx:
            return None
        out[part] = mx.contiguous(mx.load(os.path.join(art, idx[k]))[k])
    mx.eval(list(out.values()))
    return out


def call(t, x, bits, IN):
    return V._dense_fused(x, t["codes"], t["codebook"], t["vq_scales"],
                          pack_bits=bits, in_features=IN if bits else None)


def timeit(fn, reps=12, M=20, warm=2):
    for _ in range(warm):
        mx.eval([fn() for _ in range(M)])
    v = []
    for _ in range(reps):
        o = [fn() for _ in range(M)]
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(o)
        v.append((time.perf_counter() - t0) / M)
    return min(v), statistics.median(v)


def main(layer, ns, do_time):
    fails = 0
    for tag, sub, bits in RUNGS:
        art = os.path.join(MODELS, sub)
        if not os.path.isdir(art):
            print(f"\n### {tag}: artifact absent, skipped")
            continue
        cfg = json.load(open(os.path.join(art, "config.json")))
        geom = next(iter((cfg.get("vq_linear") or {}).values()), {})
        print(f"\n### {tag}  d={geom.get('dim')} K={geom.get('k')} "
              f"G={geom.get('group')} pack_bits={geom.get('pack_bits')}")
        for proj in PROJS:
            t = load(art, layer, proj)
            if t is None:
                print(f"    {proj}: not VQ in this rung, skipped")
                continue
            OUT = t["codes"].shape[0]
            NGRP = t["vq_scales"].shape[1]
            IN = NGRP * geom["group"]
            for N in ns:
                mx.random.seed(N)
                x = mx.random.normal((N, IN)).astype(mx.float16)
                mx.eval(x)
                V._DENSE_DEVX = False
                ref = call(t, x, bits, IN)
                mx.eval(ref)
                V._DENSE_DEVX = True
                got = call(t, x, bits, IN)
                mx.eval(got)
                ok = bool(mx.array_equal(ref, got))
                if not ok:
                    fails += 1
                    print(f"    {proj} N={N}: *** BIT-IDENTITY FAILED ***")
            if fails:
                continue
            print(f"    {proj} OUT={OUT} IN={IN} NGRP={NGRP}: "
                  f"bit-identical at N={list(ns)}")
            if do_time:
                x = mx.random.normal((1, IN)).astype(mx.float16)
                mx.eval(x)
                V._DENSE_DEVX = False
                a = timeit(lambda: call(t, x, bits, IN))
                V._DENSE_DEVX = True
                b = timeit(lambda: call(t, x, bits, IN))
                print(f"        N=1 us/dispatch  off {a[0]*1e6:7.1f}/"
                      f"{a[1]*1e6:7.1f}  on {b[0]*1e6:7.1f}/{b[1]*1e6:7.1f}"
                      f"   {a[0]/b[0]:.2f}x/{a[1]/b[1]:.2f}x")
            del t
            mx.clear_cache()
    print()
    if fails:
        sys.exit(f"FAIL: {fails} bit-identity failures")
    print("PASS: dense devx is bit-identical on every rung on disk")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--no-time", action="store_true")
    a = ap.parse_args()
    main(a.layer, (1, 5, 20), not a.no_time)
