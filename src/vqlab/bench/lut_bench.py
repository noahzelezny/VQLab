"""lut-bench: what does the codebook LUT read cost the decode expert kernel?

R3-5 (LUT -> ALU, rank-r structured codebooks) and R3-6 (two-level residual
codebook) are LOSSY ideas that need refits. This is their cheapest
falsifier: take the EXACT kernel the runtime dispatches at decode for a
geometry (resolved through vq_switch._fused itself, so the name, source,
template, grid and threadgroup are the shipped ones), rewrite ONLY the
codebook lookup `cb[c]`, and time stock vs arms on random tensors at decode
shapes. Codebook CONTENTS do not change the timing, so random tables are
enough; the arms are not numerically meaningful and are never compared for
values (the stock arm IS checked bit-equal to vq_switch._fused).

Arms (each replaces `cb[c]` and the stock codebook staging):
  stock   the shipped kernel, unchanged
  free    no table at all: cb[c] = half4(c). One convert keeps the code fetch
          live; this is the BOUND -- no structured codebook can beat it.
  rank1   cb[c] = a[c]*b0            (int8 K-table in threadgroup, b in regs)
  rank2   cb[c] = a0[c]*b0 + a1[c]*b1
  outer   cb[c] = U[c>>LO] * V[c&m]  (two 2^(BITS/2)-entry half4 tables)
  resid2  cb[c] = U[c>>LO] + V[c&m]  (R3-6: two-level residual, both tables
          threadgroup-resident; code bits split hi/lo)

Shape: one "layer" = gate_proj (OUT=moe_inter, IN=hidden) feeding down_proj
(OUT=hidden, IN=moe_inter), chained so dispatches serialise like decode;
--layers layers per timed eval, each layer picking --top-k fresh random
experts out of --experts (so the working set exceeds the SLC, like decode).

Each arm runs in its OWN process (rule III), arms interleaved for --rounds
rounds; the report is median us per layer and the ratio to stock.

    PYTHONPATH=src python -m vqlab.cli lut-bench --K 8192 --bits 13
    PYTHONPATH=src python -m vqlab.cli lut-bench --K 2048 --bits 11 --rounds 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ARMS = ("stock", "free", "rank1", "rank2", "outer", "resid2")


def _replace_lookups(src: str) -> str:
    """cb[<expr>] -> LUT(<expr>), bracket-balanced."""
    out, i = [], 0
    while True:
        k = src.find("cb[", i)
        if k < 0:
            out.append(src[i:])
            return "".join(out)
        if k > 0 and (src[k - 1].isalnum() or src[k - 1] == "_"):
            out.append(src[i:k + 3]); i = k + 3
            continue
        out.append(src[i:k] + "LUT(")
        depth, j = 1, k + 3
        while depth:
            depth += {"[": 1, "]": -1}.get(src[j], 0)
            j += 1
        out.append(src[k + 3:j - 1] + ")")
        i = j


def _setup(arm: str, K: int, bits: int) -> str:
    lo = bits // 2
    hi = bits - lo
    if arm == "free":
        return "    #define LUT(c) half4((half)(c))\n"
    if arm in ("rank1", "rank2"):
        R = 1 if arm == "rank1" else 2
        s = (f"    threadgroup char ta[{K * R}];\n"
             f"    const device half4* tb = (const device half4*)codebook;\n"
             f"    const device char* tsrc = (const device char*)(tb + {R});\n"
             f"    for (uint i = lid; i < {K * R}u; i += tgsize) ta[i] = tsrc[i];\n"
             f"    const float4 b0 = float4(tb[0]);\n")
        if R == 1:
            return s + "    #define LUT(c) (float(ta[(c)]) * b0)\n"
        return (s + "    const float4 b1 = float4(tb[1]);\n"
                "    #define LUT(c) (float(ta[(c)*2]) * b0 + float(ta[(c)*2+1]) * b1)\n")
    if arm in ("outer", "resid2"):
        op = "*" if arm == "outer" else "+"
        return (f"    threadgroup half4 tu[{1 << hi}];\n"
                f"    threadgroup half4 tv[{1 << lo}];\n"
                f"    const device half4* tsrc = (const device half4*)codebook;\n"
                f"    for (uint i = lid; i < {1 << hi}u; i += tgsize) tu[i] = tsrc[i];\n"
                f"    for (uint i = lid; i < {1 << lo}u; i += tgsize) tv[i] = tsrc[{1 << hi} + i];\n"
                f"    #define LUT(c) (tu[(c) >> {lo}] {op} tv[(c) & {(1 << lo) - 1}u])\n")
    raise ValueError(arm)


_STAGE_TG = ("    threadgroup half4 cb[MAX_K];\n",
             "    const device half4* cbg = (const device half4*)codebook;\n"
             "    for (uint i = lid; i < (uint)K; i += tgsize)\n"
             "        cb[i] = cbg[i];\n")
_STAGE_DEV = ("    const device half4* cb = (const device half4*)codebook;\n",)


def arm_source(src: str, arm: str, K: int, bits: int) -> str:
    if arm == "stock":
        return src
    for s in _STAGE_TG + _STAGE_DEV:
        src = src.replace(s, "")
    b = src.find("    threadgroup_barrier(")
    if b < 0:
        raise RuntimeError("no threadgroup_barrier to anchor the table setup")
    src = src[:b] + _setup(arm, K, bits) + src[b:]
    src = _replace_lookups(src)
    if "cb[" in src.replace("tb[", "") or " cb " in src:
        raise RuntimeError("a stock codebook reference survived the rewrite")
    return src


def _table(arm: str, K: int, bits: int, mx):
    lo = bits // 2
    if arm == "stock":
        return (mx.random.normal((K, 4)) * 0.05).astype(mx.float16)
    if arm == "free":
        return mx.zeros((4,), dtype=mx.float16)
    if arm in ("rank1", "rank2"):
        R = 1 if arm == "rank1" else 2
        b = (mx.random.normal((4 * R,)) * 0.01).astype(mx.float16)
        a = mx.random.randint(-127, 128, (K * R,)).astype(mx.int8)
        return mx.concatenate([b, mx.view(a, mx.float16)])
    n = (1 << (bits - lo)) + (1 << lo)
    return (mx.random.normal((n * 4,)) * 0.05).astype(mx.float16)


def child(a) -> dict:
    import mlx.core as mx
    from vqlab.runtime import vq_switch as vs
    mx.random.seed(1234)
    H, I, E, B, K = a.hidden, a.inter, a.experts, a.bits, a.K
    G = a.group
    dt = mx.bfloat16

    def proj(OUT, IN):
        wpr = (IN // 4 + 31) // 32 * B
        codes = mx.random.randint(0, 2**31 - 1, (E, OUT, wpr)).astype(mx.uint32)
        scales = (mx.random.uniform(0.5, 1.5, (E, OUT, IN // G))).astype(mx.float16)
        return codes, scales

    gate = proj(I, H)
    down = proj(H, I)
    stock_cb = (mx.random.normal((K, 4)) * 0.05).astype(mx.float16)
    eids = [mx.array(sorted(__import__("random").Random(1234 + l).sample(range(E), a.top_k)),
                     dtype=mx.uint32) for l in range(a.layers)]
    x0 = (mx.random.normal((a.top_k, H)) * 0.1).astype(dt)
    mx.eval(gate, down, stock_cb, x0, *eids)

    # resolve the SHIPPED plan for each projection through the runtime itself
    plans = {}
    for nm, (codes, scales), xin in (("gate", gate, x0),
                                    ("down", down, mx.zeros((a.top_k, I), dt))):
        y = vs._fused(xin, eids[0], codes, stock_cb, scales, pack_bits=B)
        mx.eval(y)
        key = [k for k in vs._KERNELS if isinstance(k, tuple) and k[0] == "plan"
               and k[3] == codes.shape][-1]
        view_u32, kern, name, src, template, grid, tg, dims, N, OUT = vs._KERNELS[key]
        plans[nm] = (name, src, template, grid, tg, dims, N, OUT, y, xin)

    table = stock_cb if a.arm == "stock" else _table(a.arm, K, B, mx)
    mx.eval(table)
    kerns = {}
    for nm, (name, src, template, grid, tg, dims, N, OUT, yref, xin) in plans.items():
        k = vs._get_kernel_spec(f"{name}_lut{a.arm}", arm_source(src, a.arm, K, B),
                                list(template))
        kerns[nm] = k
        if a.arm == "stock":
            codes, scales = gate if nm == "gate" else down
            (y,) = k(inputs=[xin, eids[0], codes, table, scales, dims], grid=grid,
                     threadgroup=tg, output_shapes=[(N, OUT)], output_dtypes=[dt])
            if not mx.array_equal(y, yref).item():
                raise SystemExit(f"stock relaunch != vq_switch._fused for {nm}")

    def run():
        x = x0
        for l in range(a.layers):
            for nm, (codes, scales) in (("gate", gate), ("down", down)):
                _, _, _, grid, tg, dims, N, OUT, _, _ = plans[nm]
                (x,) = kerns[nm](inputs=[x, eids[l], codes, table, scales, dims],
                                 grid=grid, threadgroup=tg,
                                 output_shapes=[(N, OUT)], output_dtypes=[dt])
        mx.eval(x)

    for _ in range(a.warmup):
        run()
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
    us = statistics.median(ts) / a.layers * 1e6
    return {"arm": a.arm, "us_per_layer": us,
            "kernels": {nm: p[0] for nm, p in plans.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--bits", type=int, default=13, help="pack_bits (K must be 2**bits)")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--json", help="write per-round results here")
    ap.add_argument("--arm", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.K != 1 << a.bits:
        raise SystemExit("--K must equal 2**--bits (random codes must all be valid)")
    if a.arm:
        print("RESULT " + json.dumps(child(a)), flush=True)
        return 0
    if "stock" not in a.arms:
        a.arms = ["stock"] + a.arms
    base = list(sys.argv[1:])
    res = {arm: [] for arm in a.arms}
    kern = None
    for r in range(a.rounds):
        for arm in a.arms:
            p = subprocess.run([sys.executable, "-m", "vqlab.bench.lut_bench", *base,
                                "--arm", arm], capture_output=True, text=True,
                               env=os.environ)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if p.returncode or not line:
                print(p.stdout, p.stderr, file=sys.stderr)
                raise SystemExit(f"arm {arm} failed")
            d = json.loads(line[0][7:])
            kern = kern or d["kernels"]
            res[arm].append(d["us_per_layer"])
            print(f"round {r} {arm:7s} {d['us_per_layer']:8.2f} us/layer", flush=True)
    print(f"\nK={a.K} bits={a.bits} N={a.top_k} pairs  kernels: {kern}")
    s = statistics.median(res["stock"])
    print(f"{'arm':8s} {'us/layer':>9s} {'min':>8s} {'max':>8s}  speed vs stock")
    for arm, v in res.items():
        m = statistics.median(v)
        print(f"{arm:8s} {m:9.2f} {min(v):8.2f} {max(v):8.2f}  {s / m:.3f}x of stock speed")
    if a.json:
        json.dump({"args": vars(a), "kernels": kern, "us_per_layer": res},
                  open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
