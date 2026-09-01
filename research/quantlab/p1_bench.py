#!/usr/bin/env python3
"""P1: box-selection benchmark for the 397B VQ fit (VQ_397B_OVERNIGHT_PLAN).

Three numbers per box:
  gemm  — fp32 GEMM rate at VQ-fit shapes (the k-means assign kernel)
  read  — sequential read of one real 397B source shard
  write — write+fsync of 4 GB to the artifact destination

Run with the box's own python (needs mlx + numpy only).
"""
import sys
import time
import os
import pathlib

import numpy as np
import mlx.core as mx

SRC = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16")
DST = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models/.p1_bench_scratch")

# --- gemm: emulate the assign step: [N,K] = [N,4]@[4,K] is tiny; the real
# cost is X@C.T at N=4M,K=1024 plus the [rows,in] reshapes. Use the actual
# dominant shape: [4M x 4] @ [4 x 1024] AND a bulk [8192x4096]@[4096x8192].
def bench_gemm():
    X = mx.random.normal((4_000_000, 4))
    C = mx.random.normal((1024, 4))
    mx.eval(X, C)
    t0 = time.time()
    for _ in range(10):
        d = X @ C.T
        mx.eval(d)
    t_assign = (time.time() - t0) / 10
    A = mx.random.normal((8192, 4096))
    B = mx.random.normal((4096, 8192))
    mx.eval(A, B)
    t0 = time.time()
    for _ in range(20):
        mx.eval(A @ B)
    t_bulk = (time.time() - t0) / 20
    tflops = 2 * 8192 * 4096 * 8192 / t_bulk / 1e12
    return t_assign, tflops


def bench_read():
    import json
    idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
    # a shard holding layer-5 experts (mid-file, no cache warmth games)
    shard = idx["model.language_model.layers.5.mlp.experts.gate_up_proj"]
    p = SRC / shard
    sz = p.stat().st_size
    t0 = time.time()
    with open(p, "rb", buffering=0) as f:
        while f.read(1 << 24):
            pass
    dt = time.time() - t0
    return sz / dt / 1e9, sz / 1e9


def bench_write():
    DST.mkdir(exist_ok=True)
    buf = os.urandom(1 << 26)  # 64 MB
    n = 64  # 4 GB
    p = DST / f"w_{os.uname().nodename}.bin"
    t0 = time.time()
    with open(p, "wb", buffering=0) as f:
        for _ in range(n):
            f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    dt = time.time() - t0
    p.unlink()
    return n * len(buf) / dt / 1e9


if __name__ == "__main__":
    host = os.uname().nodename
    t_assign, tflops = bench_gemm()
    rd, shard_gb = bench_read()
    wr = bench_write()
    # projection: per gate_up tensor ~ assign over 1.07e9 subvecs
    # = 268 chunks of 4M -> t_assign*268; x2 tensors x30 layers (+kmeans ~15%)
    fit_h = (t_assign * 268 * 2 * 30 * 1.15) / 3600
    io_h = (390 / rd + 445 / wr) / 3600
    print(f"HOST {host}")
    print(f"  gemm: assign-chunk {t_assign*1000:.0f} ms   bulk {tflops:.1f} TFLOPS")
    print(f"  read: {rd:.2f} GB/s (shard {shard_gb:.1f} GB)")
    print(f"  write+fsync: {wr:.2f} GB/s")
    print(f"  PROJECTED: fit {fit_h:.1f} h + io {io_h:.1f} h = {fit_h+io_h:.1f} h total")
