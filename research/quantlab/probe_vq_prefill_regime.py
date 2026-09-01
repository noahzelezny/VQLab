#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""VQ-PF1 probe: which regime does the VQ prefill path win, and where does
the end-to-end 9x live?

vq_switch.py's header claims decode+GEMM prefill is 1.21-1.28x gather_qmm
(m1c_prefill_bench, 8192 tokens x top_k 8 through ONE tensor). The 08-16
task sweep measured ~90 s/block vs spicy's ~9.6 s/block on the same
289,598-token workload. Both are real; this probe finds the axis that
separates them.

It loads ONE REAL expert tensor (layer 3 gate_proj) out of the shipped
2.4bpw artifact and sweeps rows-per-expert -- the only quantity that
changes between the microbench and the scorer -- timing decode and GEMM
separately against an affine gather_qmm baseline of identical shape.

  ./probe_vq_prefill_regime.py                 # default sweep
  ./probe_vq_prefill_regime.py --tokens 8192   # single point
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from safetensors import safe_open

import vq_switch

ART = ("/Volumes/Thunderbay SSD/Exo Models/"
       "Qwen3.5-397B-A17B-VQ-2.4bpw/model-00002-of-00027.safetensors")
PREFIX = "language_model.model.layers.3.mlp.switch_mlp.gate_proj"


def bench(fn, warmup=1, reps=3):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def load_vq():
    with safe_open(ART, framework="numpy") as f:
        codes = f.get_tensor(f"{PREFIX}.codes")
        cb = f.get_tensor(f"{PREFIX}.codebook")
        sc = f.get_tensor(f"{PREFIX}.vq_scales")
    return (mx.array(codes), mx.array(cb).astype(mx.float16),
            mx.array(sc).astype(mx.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=None,
                    help="single point instead of the sweep")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=None,
                    help="override _DECODE_CHUNK")
    args = ap.parse_args()

    codes, cb, sc = load_vq()
    mx.eval(codes, cb, sc)
    E, OUT, NSUB = codes.shape
    K, D = cb.shape
    IN = NSUB * D
    G = IN // sc.shape[2]
    print(f"tensor {PREFIX}")
    print(f"  E={E} OUT={OUT} IN={IN} d={D} K={K} group={G} "
          f"codes.dtype={codes.dtype}")

    if args.chunk:
        vq_switch._DECODE_CHUNK = args.chunk
    if vq_switch._DECODE_CHUNK is None:
        vq_switch._DECODE_CHUNK = vq_switch._default_decode_chunk()
    chunk = vq_switch._DECODE_CHUNK
    print(f"  _DECODE_CHUNK = {chunk}  ({-(-E // chunk)} chunks for {E} experts)")

    # affine 2-bit baseline of identical shape (what spicy's gather_qmm eats)
    Wb = mx.random.normal((E, OUT, IN)).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(Wb, group_size=64, bits=2)
    mx.eval(qw, qs, qb)
    del Wb
    mx.clear_cache()

    # decode-only cost for the whole tensor, independent of token count
    t_dec_full = 0.0
    for c0 in range(0, E, chunk):
        eids = mx.array(np.arange(c0, min(c0 + chunk, E), dtype=np.uint32))
        t_dec_full += bench(lambda e=eids: vq_switch._decode_chunk(
            codes, cb, sc, e, in_features=IN))
    dense_gib = E * OUT * IN * 2 / 2**30
    print(f"  decode ALL {E} experts: {t_dec_full*1e3:8.1f} ms  "
          f"({dense_gib:.2f} GiB fp16 materialized)")

    token_grid = ([args.tokens] if args.tokens
                  else [1024, 2048, 4096, 9050, 18100, 36200, 72400])
    rng = np.random.default_rng(3)
    print()
    print(f"{'tokens':>8} {'rows/exp':>9} {'gather_qmm':>11} {'vq_prefill':>11} "
          f"{'ratio':>7} {'decode':>9} {'gemm':>9} {'dec%':>6}")
    for T in token_grid:
        idx_np = rng.integers(0, E, (T, args.top_k)).astype(np.uint32)
        idx_np = np.sort(idx_np, axis=1)
        xtok = mx.random.normal((T, IN)).astype(mx.float16)
        xq = xtok[:, None, None, :].astype(mx.bfloat16)
        idx = mx.array(idx_np)
        mx.eval(xtok, xq, idx)

        t_base = bench(lambda: mx.gather_qmm(
            xq, qw, qs, qb, rhs_indices=idx, transpose=True,
            group_size=64, bits=2, sorted_indices=True))

        flat = idx_np.reshape(-1)
        order = np.argsort(flat, kind="stable")
        e_sorted = flat[order]
        tok_of = np.repeat(np.arange(T), args.top_k)[order]
        xf = xtok[mx.array(tok_of.astype(np.uint32))]
        mx.eval(xf)

        t_vq = bench(lambda: vq_switch._prefill(
            xf, e_sorted, codes, cb, sc, in_features=IN))

        # GEMM-only: same padded batched matmul against a PRE-decoded chunk,
        # so the difference from t_vq is decode + its eval barrier.
        t_gemm = t_vq - t_dec_full
        rows = len(e_sorted) / E
        print(f"{T:8d} {rows:9.0f} {t_base*1e3:10.1f}m {t_vq*1e3:10.1f}m "
              f"{t_base/t_vq:6.2f}x {t_dec_full*1e3:8.1f}m {t_gemm*1e3:8.1f}m "
              f"{100*t_dec_full/t_vq:5.1f}%")
        del xtok, xq, idx, xf
        mx.clear_cache()


if __name__ == "__main__":
    main()
