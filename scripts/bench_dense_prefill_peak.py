#!/usr/bin/env python
"""PREFILL PEAK vs RESIDENT on the dense large-N path (2026-09-03).

Noah's measurement: 27B 3.9bpw, 2048-token prompt, peak 18.7 G vs 11.6 G
active. The suspect is vq_dense's large-N path -- above the fused-kernel N
cutoff every VQLinear materialises a decoded fp16 weight, and mlx's lazy
graph keeps many layers' worth of them alive at once.

This isolates that claim WITHOUT a model load: it builds a chain of REAL
VQLinear modules from the artifact's own tensors (gate_proj 5120->17408 then
down_proj 17408->5120, repeated), runs one prefill-width batch through it,
and reports peak, active-at-end, and wall time per arm.

  arms (VQ_DENSE_DECODE_TILE_MB)
    0     the pre-arc path: decode the whole weight, no forced eval.
          Unbounded peak -- grows with how deep the graph is built.
    512   one tile (178 MB at these shapes) + a forced eval per linear:
          caps the live set at ONE layer's transient.
    64    real row tiling: transient bounded to ~64 MB per linear.
    16    tighter still, to price the sync/tiling overhead.

Bit-identity across every arm is asserted against arm 0 before any number
is reported -- tiling is along OUT only, so no reduction is reordered.

Run:  PYTHONPATH=src python scripts/bench_dense_prefill_peak.py --layers 8
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx

from vqlab import vq_dense as VD

ART_DEFAULT = os.path.expanduser(
    "~/.exo/models/TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw")
GiB = float(1 << 30)
MiB = float(1 << 20)


def load(art, layer, proj):
    idx = json.load(open(os.path.join(
        art, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for part in ("codes", "codebook", "vq_scales"):
        k = f"language_model.model.layers.{layer}.mlp.{proj}.{part}"
        out[part] = mx.contiguous(mx.load(os.path.join(art, idx[k]))[k])
    mx.eval(list(out.values()))
    return out


def build(art, nlayers, bits, group):
    """A chain of real VQLinears: (gate, down) x nlayers, shapes chained."""
    mods = []
    for i in range(nlayers):
        for proj in ("gate_proj", "down_proj"):
            t = load(art, 20, proj)
            IN = t["vq_scales"].shape[1] * group
            mods.append(VD.VQLinear(
                t["codes"], t["codebook"], t["vq_scales"], group_size=group,
                pack_bits=bits, in_features=IN if bits else None))
    return mods


def run(mods, x):
    h = x
    for m in mods:
        h = m(h)
    mx.eval(h)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART_DEFAULT)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--arms", default="0,512,64,16")
    a = ap.parse_args()
    if not os.path.isdir(a.art):
        sys.exit(f"artifact not readable: {a.art}")
    cfg = json.load(open(os.path.join(a.art, "config.json")))
    geom = next(iter((cfg.get("vq_linear") or {}).values()), {})
    bits, group = geom.get("pack_bits") or 0, geom.get("group", 64)

    mods = build(a.art, a.layers, bits, group)
    resident = mx.get_active_memory() / GiB
    print(f"artifact {os.path.basename(a.art)}  d={geom.get('dim')} "
          f"K={geom.get('k')} pack_bits={bits}")
    print(f"chain: {len(mods)} real VQLinears ({a.layers} mlp layer-pairs), "
          f"N={a.tokens}")
    print(f"weights resident after build: {resident:.3f} GiB\n")

    IN = mods[0].input_dims
    mx.random.seed(0)
    x = mx.random.normal((a.tokens, IN)).astype(mx.float16)
    mx.eval(x)

    arms = [float(s) for s in a.arms.split(",")]
    ref = None
    print(f"{'tile_MB':>8s} {'peak':>9s} {'peak-resident':>14s} "
          f"{'active_end':>11s} {'wall_s':>8s} {'ident':>7s}")
    for mb in arms:
        VD._DENSE_DECODE_TILE_MB = mb
        run(mods, x[:8])                      # warm / JIT
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        y = run(mods, x)
        wall = time.perf_counter() - t0
        peak = mx.get_peak_memory() / GiB
        act = mx.get_active_memory() / GiB
        if ref is None:
            ref, ident = y, "ref"
        else:
            # BIT patterns, not array_equal: a deep unnormalised chain of
            # random matmuls saturates fp16, and array_equal reports NaN !=
            # NaN even when the two arms produced the identical bit pattern.
            # Comparing the uint16 views is both nan-proof and strictly
            # stronger (it would catch a differing NaN payload too).
            ident = "YES" if bool(mx.array_equal(
                ref.view(mx.uint16), y.view(mx.uint16))) else "*** NO ***"
        print(f"{mb:8.0f} {peak:8.3f}G {peak - resident:13.3f}G "
              f"{act:10.3f}G {wall:8.3f} {ident:>7s}")
        if ident.startswith("***"):
            sys.exit("BIT-IDENTITY FAILED -- numbers above are not comparable")
        del y
        mx.clear_cache()
    print("\nnote: peak here counts only this chain, so peak-resident is the "
          "\ntransient the prefill path adds on top of the weights.")


if __name__ == "__main__":
    main()
