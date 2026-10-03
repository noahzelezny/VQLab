"""vqlab kernel-truth-moe — the MoE twin of kernel-truth (devlist #15).

Which MoE expert path is closer to EXACT: the small-N fused decode kernel
(N = token-expert pairs <= VQ_FUSED_MAX_N, what generation runs) or the
large-N gemmseg2 prefill kernel (what prefill and the KL gate run)? The
reference is the float64 reconstruction of each expert's W from its codes,
codebook and scales (packed codes unpacked with vq_pack.unpack, the reader
the Metal kernels must agree with), applied to the SAME bf16-rounded inputs
the kernels get (see kernel-truth for why rounding the reference input
matters). Rows are independent, so the fused run's tokens are the leading
rows of the prefill run: identical inputs, two kernels.

    vqlab kernel-truth-moe --synthetic                 # DeepSeek-V4 shapes, K2048 and K4096
    vqlab kernel-truth-moe --artifact <dir> --family deepseek_v4 [--modules ...]

Answers a question the KL gate cannot: the gate scores at prefill N, so a
decode-kernel numerics difference is invisible to it (F137/F164).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import mlx.core as mx

from vqlab import _layout  # noqa: F401  one module object per name
from vqlab.runtime import vq_pack
from vqlab.runtime import vq_switch as VS


def _bf16(a):
    return np.array(mx.array(a).astype(mx.bfloat16).astype(mx.float32))


def _weights(mod, experts):
    """{e: float64 W_e [OUT, IN]} from the module's own tensors."""
    codes = np.array(mod.codes)
    cb = np.array(mod.codebook.astype(mx.float32)).astype(np.float64)
    sc = np.array(mod.vq_scales.astype(mx.float32)).astype(np.float64)
    D, IN = cb.shape[1], mod.input_dims
    out = {}
    for e in experts:
        c = codes[e:e + 1]
        if mod.pack_bits:
            c = vq_pack.unpack(c, IN // D, mod.pack_bits)
        c = c[0].astype(np.int64)                       # [OUT, NSUB]
        W = cb[c].reshape(c.shape[0], -1)               # [OUT, IN]
        out[e] = W * np.repeat(sc[e], IN // sc.shape[-1], axis=1)
    return out


def compare(name, mod, seeds=3, n_dec=8, topk=6, n_experts=8):
    """Print and return [(winner, ratio)] for one VQSwitchLinear."""
    E = mod.codes.shape[0]
    IN = mod.input_dims
    pool = np.arange(min(E, n_experts))
    Ws = _weights(mod, pool)
    pairs_big = VS.VQ_FUSED_MAX_N // topk + 16           # just above the fused gate
    res = []
    for seed in range(seeds):
        r = np.random.default_rng(1234 + seed)
        x = _bf16(r.standard_normal((pairs_big, IN)).astype("f4") * 0.5)
        idx = np.stack([r.choice(pool, topk, replace=False) for _ in range(pairs_big)]).astype(np.uint32)
        ref = np.stack([np.stack([x[t].astype(np.float64) @ Ws[e].T for e in idx[t]])
                        for t in range(n_dec)])          # [n_dec, topk, OUT]

        def run(n):
            y = mod(mx.array(x[:n])[:, None, None, :].astype(mx.bfloat16), mx.array(idx[:n]))
            mx.eval(y)
            return np.array(y.astype(mx.float32)).astype(np.float64)[:n_dec].reshape(n_dec, topk, -1)

        dec, pre = run(n_dec), run(pairs_big)
        ed = float(np.sqrt(((dec - ref) ** 2).mean()))
        ep = float(np.sqrt(((pre - ref) ** 2).mean()))
        w = "DECODE" if ed < ep else "PREFILL"
        res.append((w, max(ed, ep) / max(min(ed, ep), 1e-30)))
        print(f"  {name:40s} seed{seed}  decode rms {ed:.4e}  prefill rms {ep:.4e}  -> {w} by {res[-1][1]:.3f}x",
              flush=True)
    return res


def synthetic(K, bits, IN=4096, OUT=2048, E=16, seed=1234):
    r = np.random.default_rng(seed)
    nsub = IN // 4
    codes = r.integers(0, K, (E, OUT, nsub)).astype(np.uint16)
    packed = vq_pack.pack(codes, bits)
    cb = (r.standard_normal((K, 4)) * 0.05).astype(np.float16)
    sc = (r.random((E, OUT, IN // 64)) * 0.5 + 0.5).astype(np.float16)
    return VS.VQSwitchLinear(mx.array(packed), mx.array(cb), mx.array(sc), group_size=64,
                             pack_bits=bits, in_features=IN)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab kernel-truth-moe", description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true", help="DeepSeek-V4 expert shapes, K2048/bits11 and K4096/bits12")
    ap.add_argument("--artifact")
    ap.add_argument("--family", default="deepseek_v4")
    ap.add_argument("--modules", default="layers.1.ffn.switch_mlp.gate_proj,layers.24.ffn.switch_mlp.up_proj,"
                                         "layers.40.ffn.switch_mlp.down_proj")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    cells = []
    if a.synthetic:
        for K, bits in ((2048, 11), (4096, 12)):
            cells += compare(f"synthetic d4/K{K} packed{bits}", synthetic(K, bits), a.seeds)
    elif a.artifact:
        from vqlab import runtime_load
        model, _ = runtime_load.load_for_family(a.family, a.artifact, lazy=True)
        want = tuple(s.strip() for s in a.modules.split(",") if s.strip())
        pick = [(n, m) for n, m in model.named_modules()
                if type(m).__name__ == "VQSwitchLinear" and n.endswith(want)]
        if not pick:
            raise SystemExit(f"FAIL: no VQSwitchLinear matched {want}")
        for n, m in pick:
            cells += compare(n.split("model.")[-1], m, a.seeds)
    else:
        ap.error("--synthetic or --artifact")
    wins = {"DECODE": sum(w == "DECODE" for w, _ in cells), "PREFILL": sum(w == "PREFILL" for w, _ in cells)}
    best = max(wins, key=wins.get)
    print(f"\nVERDICT over {len(cells)} cells: {wins}   median ratio "
          f"{float(np.median([r for _, r in cells])):.3f}x")
    print(f"UNANIMOUS: the {best} path is closer to exact on every cell." if wins[best] == len(cells)
          else "SPLIT: the verdict depends on the module or the input; do NOT quote a single ratio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
