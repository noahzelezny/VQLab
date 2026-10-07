"""vqlab kernel-truth — which VQ path is closer to EXACT: fused or fallback?

WHY THIS EXISTS. F145 measured up to 1.277 logits of disagreement between
the fused decode kernel and the `_decode_matmul` (wdec + GEMM) fallback, and
which one an artifact uses flips at a batch size that differs between rungs
(plain 12, packed d2 72, packed d4 32). A DIVERGENCE figure says two
roundings disagree; it does NOT say which is wrong. Reading a divergence as
an error is exactly the mistake F138 corrected on the "8 ULP" question,
where the supposedly-degraded path turned out to be the BETTER-rounded one
and had been shipped disabled for weeks because nobody asked.

THE REFERENCE IS EXACT, NOT ANOTHER APPROXIMATION. VQ reconstruction of W
from codes/codebook/scales is a lookup and a multiply, so it can be done in
float64 with no error of its own; y = x @ W.T in float64 is then the true
answer both kernels are approximating. No teacher model is required, which
matters because this family's bf16 teacher is gone.

THE INPUT IS ROUNDED TO bf16 FIRST, deliberately. The kernels receive bf16;
if the reference is given the unrounded float32 input, bf16 input rounding
contributes an error COMMON to both paths and shrinks the measured gap
between them. Measured on the 4.8: 1.24x with the naive reference, 1.43x
with the honest one.

Rows are independent, so feeding N inside the fused gate and N above it,
sharing the same leading rows, compares the two paths on identical inputs.

    vqlab kernel-truth --artifact <dir> [--n-fused 8 --n-fallback 16]

SWITCH MODE (MoE prefill, 2026-10-07). `--switch-flag VQ_GEMMSEG_X` compares
the VQSwitchLinear prefill kernel with that module flag OFF vs ON, both
against the same exact float64 reference, at `--switch-tokens` tokens routed
top-8 over the first 16 experts (N pairs above VQ_FUSED_MAX_N, so gemmseg2
runs). The artifact is loaded through its OWN model.py (mlx_lm.load), so a
pinned "-kern" twin measures the branch runtime it carries.

    vqlab kernel-truth --artifact <pin> --switch-flag VQ_GEMMSEG_ACCS
"""
from __future__ import annotations

import argparse
import sys
import pathlib

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
from vqlab import runtime_load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--family", default="qwen3_5")
    ap.add_argument("--n-fused", type=int, default=8,
                    help="rows INSIDE the fused gate (plain gate is 12)")
    ap.add_argument("--n-fallback", type=int, default=16,
                    help="rows ABOVE the gate, so _decode_matmul is taken")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modules", default="layers.1.mlp.gate_proj,"
                    "layers.20.mlp.up_proj,layers.40.mlp.down_proj,"
                    "layers.63.mlp.gate_proj",
                    help="comma-separated name suffixes; spread across depth "
                         "and across projections so one layer's luck cannot "
                         "carry the verdict")
    ap.add_argument("--switch-flag", default=None,
                    help="MoE mode: vq_switch module flag name to A/B, e.g. "
                         "VQ_GEMMSEG_ACCS (module attr _GEMMSEG_ACCS)")
    ap.add_argument("--switch-tokens", type=int, default=1024)
    ap.add_argument("--switch-modules", default="layers.0.mlp.switch_mlp.gate_proj,"
                    "layers.13.mlp.switch_mlp.up_proj,"
                    "layers.26.mlp.switch_mlp.down_proj,"
                    "layers.39.mlp.switch_mlp.gate_proj")
    a = ap.parse_args()
    if a.switch_flag:
        return switch_main(a)

    model, _ = runtime_load.load_for_family(a.family, a.artifact, lazy=True)
    want = tuple(s.strip() for s in a.modules.split(",") if s.strip())
    pick = [(n, m) for n, m in model.named_modules()
            if type(m).__name__ == "VQLinear" and n.endswith(want)]
    if not pick:
        raise SystemExit(f"FAIL: no VQLinear matched {want}")

    def bf16_exact(arr):
        return np.array(mx.array(arr).astype(mx.bfloat16).astype(mx.float32))

    wins = {"FUSED": 0, "FALLBACK": 0}
    ratios = []
    print(f"\n{len(pick)} modules, {a.seeds} seeds, N={a.n_fused} (fused) vs "
          f"N={a.n_fallback} (fallback)\n")
    for mname, mod in pick:
        IN = mod.input_dims
        codes = np.array(mod.codes)
        cb = np.array(mod.codebook.astype(mx.float32)).astype(np.float64)
        sc = np.array(mod.vq_scales.astype(mx.float32)).astype(np.float64)
        OUT, NSUB = codes.shape
        W = cb[codes.astype(np.int64)].reshape(OUT, NSUB * cb.shape[1])
        srow = sc.reshape(OUT, -1)
        W = W * np.repeat(srow, IN // srow.shape[1], axis=1)

        def run(xnp):
            y = mod(mx.array(xnp)[None].astype(mx.bfloat16))
            mx.eval(y)
            return np.array(y[0].astype(mx.float32)).astype(np.float64)

        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            xa = bf16_exact(rng.standard_normal((a.n_fused, IN)).astype("f4"))
            pad = bf16_exact(rng.standard_normal(
                (a.n_fallback - a.n_fused, IN)).astype("f4"))
            ref = xa.astype(np.float64) @ W.T
            fused = run(xa)
            fallb = run(np.concatenate([xa, pad], 0))[: a.n_fused]
            ef = float(np.sqrt(((fused - ref) ** 2).mean()))
            eb = float(np.sqrt(((fallb - ref) ** 2).mean()))
            w = "FUSED" if ef < eb else "FALLBACK"
            wins[w] += 1
            ratios.append(max(ef, eb) / min(ef, eb))
            print(f"  {mname.split('model.')[-1]:34s} seed{seed}  "
                  f"fused rms {ef:.4e}  fallback rms {eb:.4e}  -> {w} "
                  f"by {ratios[-1]:.3f}x")

    n = sum(wins.values())
    best = max(wins, key=wins.get)
    print(f"\nVERDICT over {n} cells: {wins}   "
          f"median ratio {float(np.median(ratios)):.3f}x")
    if wins[best] == n:
        print(f"UNANIMOUS: the {best} path is closer to exact on every cell.")
    else:
        print("SPLIT: the verdict depends on the module or the input; do NOT "
              "quote a single ratio from this run.")
    return 0


def switch_main(a) -> int:
    """OFF vs ON for one vq_switch flag, each against exact float64."""
    from mlx_lm import load
    from vqlab.runtime import vq_pack
    model, _ = load(a.artifact)
    want = tuple(x.strip() for x in a.switch_modules.split(",") if x.strip())
    pick = [(n, m) for n, m in model.named_modules()
            if type(m).__name__ == "VQSwitchLinear" and n.endswith(want)]
    if not pick:
        raise SystemExit(f"FAIL: no VQSwitchLinear matched {want}")
    # bundle model.py modules are not in sys.modules; reach the globals
    vqg = type(pick[0][1]).__call__.__globals__
    attr = "_" + a.switch_flag[3:] if a.switch_flag.startswith("VQ_") else a.switch_flag
    if attr not in vqg:
        raise SystemExit(f"FAIL: runtime has no {attr} (wrong model.py?)")
    E_USE, TOPK, T = 16, 8, a.switch_tokens

    def bf16_exact(arr):
        return np.array(mx.array(arr).astype(mx.bfloat16).astype(mx.float32))

    wins = {"OFF": 0, "ON": 0, "TIE": 0}
    rel = []
    print(f"\n{len(pick)} modules, {a.seeds} seeds, T={T} x top{TOPK} over "
          f"{E_USE} experts (N={T*TOPK} pairs), flag {attr} OFF vs ON\n")
    for mname, mod in pick:
        IN, OUT = mod.input_dims, mod.output_dims
        cb = np.array(mod.codebook.astype(mx.float32)).astype(np.float64)
        D = cb.shape[1]; NSUB = IN // D
        sc = np.array(mod.vq_scales.astype(mx.float32)).astype(np.float64)
        Ws = []
        for e in range(E_USE):
            c = np.array(mod.codes[e])
            if mod.pack_bits:
                c = vq_pack.unpack(c[None], NSUB, mod.pack_bits)[0]
            W = cb[c.astype(np.int64)].reshape(OUT, IN)
            s_e = sc[e].reshape(OUT, -1)
            Ws.append(W * np.repeat(s_e, IN // s_e.shape[1], axis=1))
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            x = bf16_exact(rng.standard_normal((T, IN)).astype("f4"))
            idx = np.stack([rng.choice(E_USE, TOPK, replace=False)
                            for _ in range(T)]).astype(np.int32)
            ref = np.stack([x.astype(np.float64) @ Ws[e].T for e in range(E_USE)])
            ref = ref[idx, np.arange(T)[:, None]]          # [T, TOPK, OUT]
            out = {}
            for flag in (False, True):
                vqg[attr] = flag
                xm = mx.broadcast_to(mx.array(x).astype(mx.bfloat16)
                                     [None, :, None, None, :], (1, T, TOPK, 1, IN))
                y = mod(xm, mx.array(idx)[None])
                mx.eval(y)
                out[flag] = np.array(y.astype(mx.float32)).astype(
                    np.float64).reshape(T, TOPK, OUT)
            vqg[attr] = False
            e0 = float(np.sqrt(((out[False] - ref) ** 2).mean()))
            e1 = float(np.sqrt(((out[True] - ref) ** 2).mean()))
            w = "TIE" if e0 == e1 else ("ON" if e1 < e0 else "OFF")
            wins[w] += 1
            rel.append(e1 / e0)
            print(f"  {mname.split('model.')[-1]:40s} seed{seed}  OFF rms "
                  f"{e0:.4e}  ON rms {e1:.4e}  ON/OFF {e1/e0:.3f}  -> {w}")
    print(f"\nVERDICT over {sum(wins.values())} cells: {wins}   median ON/OFF "
          f"{float(np.median(rel)):.3f}")
    if wins["TIE"] == sum(wins.values()):
        print("ALL TIE: outputs identical -- check the flag reached the kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
