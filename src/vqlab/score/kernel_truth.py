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


def _num(v):
    """'512' -> 512 so a numeric knob compares as a number; strings pass."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


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
                         "VQ_GEMMSEG_ACCS (module attr _GEMMSEG_ACCS), or a "
                        "numeric knob such as VQ_FUSED_MAX_N with --switch-value 512")
    ap.add_argument("--switch-value", default=None,
                    help="value the flag takes when ON (default True); for a "
                         "string-valued flag, e.g. --switch-flag "
                         "VQ_PREFILL_EXPAND --switch-value fp16")
    ap.add_argument("--switch-tokens", type=int, default=1024)
    ap.add_argument("--acts-corpus", default=None,
                    help="switch mode: replace the random-normal inputs with "
                         "REAL activations -- the (x, routing indices) each "
                         "picked module receives in ONE forward over the "
                         "first --switch-tokens tokens of this corpus (F202: "
                         "random inputs saw a uniform 0.8%% where code text "
                         "moved KL by 22 mnats)")
    ap.add_argument("--acts-offset", type=int, default=0,
                    help="start token of the --acts-corpus window")
    ap.add_argument("--arm", action="append", default=[],
                    help="with --acts-corpus: NAME=FLAG:VAL[,FLAG:VAL] -- "
                         "one runtime configuration, every arm scored against "
                         "the same float64 reference (repeatable; 'base' = "
                         "the runtime's own values is always run first). "
                         "FLAG is a VQ_* name or module attr.")
    ap.add_argument("--switch-modules", default="layers.0.mlp.switch_mlp.gate_proj,"
                    "layers.13.mlp.switch_mlp.up_proj,"
                    "layers.26.mlp.switch_mlp.down_proj,"
                    "layers.39.mlp.switch_mlp.gate_proj")
    a = ap.parse_args()
    if a.acts_corpus:
        return acts_main(a)
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
        OUT = codes.shape[0]
        NSUB = IN // cb.shape[1]
        if getattr(mod, "pack_bits", 0):
            # packed dense rows ([OUT, WPR] uint32): unpack first, or the
            # words index the codebook as codes (crashed on 27B 3.9, F203)
            from vqlab.runtime import vq_pack
            codes = vq_pack.unpack(codes[None], NSUB, mod.pack_bits)[0]
        codes = codes.reshape(OUT, NSUB)
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


def _parse_arm(spec):
    """'name=VQ_A:1,VQ_B:512' -> ('name', {'VQ_A': 1, 'VQ_B': 512})."""
    name, _, body = spec.partition("=")
    kv = {}
    for item in filter(None, (s.strip() for s in body.split(","))):
        k, _, v = item.partition(":")
        kv[k] = {"": True, "true": True, "false": False}.get(v, _num(v))
    return name, kv


def _attr(vqg, flag):
    attr = flag if flag in vqg else (
        "_" + flag[3:] if flag.startswith("VQ_") else flag)
    if attr not in vqg:
        raise SystemExit(f"FAIL: runtime has no {attr} (wrong model.py?)")
    return attr


def acts_main(a) -> int:
    """Real-activation switch mode (F202): capture what each picked
    VQSwitchLinear actually receives on a corpus window, then score every
    --arm against the exact float64 product on those inputs.

    Also prints three host SIMULATIONS (float64 numpy, no kernel) that
    isolate one rounding each on the same inputs, every one followed by the
    kernels' own output rounding (fp32 -> fp16 -> bf16):
      sim:out       the output rounding alone
      sim:x16       + the fp16 cast of the bf16 input
      sim:x16+w16   + fp16 rounding of each staged weight s*cb (gemmseg2's
                    wtT; the fused walk never rounds the product)
    A kernel arm far above sim:x16+w16 carries an error the simulation does
    not model (accumulation / matrix-unit precision)."""
    from mlx_lm import load
    from vqlab.runtime import vq_pack
    model, tok = load(a.artifact)
    want = tuple(x.strip() for x in a.switch_modules.split(",") if x.strip())
    pick = [(n, m) for n, m in model.named_modules()
            if type(m).__name__ == "VQSwitchLinear" and n.endswith(want)]
    if not pick:
        raise SystemExit(f"FAIL: no VQSwitchLinear matched {want}")
    vqg = type(pick[0][1]).__call__.__globals__
    arms = [("base", {})] + [_parse_arm(s) for s in a.arm]

    # ---- capture: one forward, every picked module records its call ------
    text = open(a.acts_corpus, encoding="utf-8").read()
    ids = tok.encode(text)[a.acts_offset:a.acts_offset + a.switch_tokens]
    cls = type(pick[0][1])
    names = {id(m): n for n, m in pick}
    rec = {}
    orig = cls.__call__

    def recording(self, x, indices, sorted_indices=False):
        if id(self) in names and id(self) not in rec:
            mx.eval(x, indices)
            rec[id(self)] = (np.array(x.astype(mx.float32)),
                             np.array(indices).astype(np.int64),
                             bool(sorted_indices))
        return orig(self, x, indices, sorted_indices)
    cls.__call__ = recording
    try:
        mx.eval(model(mx.array(ids)[None]))
    finally:
        cls.__call__ = orig
    print(f"\ncaptured {len(rec)}/{len(pick)} modules from "
          f"{pathlib.Path(a.acts_corpus).name} tokens "
          f"[{a.acts_offset}, {a.acts_offset + len(ids)}), one forward "
          f"(N = {len(ids)} x top-k pairs per module)")
    print("arms: " + "; ".join(f"{n}={kv or 'runtime defaults'}" for n, kv in arms))

    def f16(v):
        return v.astype(np.float16).astype(np.float64)

    # The kernels' output store. bf16-I/O runtimes (profile v2) round fp32
    # straight to bf16 ONCE; v1.5 stores fp16 and the host casts to bf16
    # (a double rounding that alone flips ~6% of outputs off the correctly
    # rounded value). Follow the runtime actually loaded.
    double = not (vqg.get("_GEMMSEG_BF16IO") and vqg.get("_DECODE_BF16IO"))
    print(f"output store modelled as "
          f"{'fp32->fp16->bf16 (v1.5)' if double else 'fp32->bf16 (bf16 I/O)'}")

    def outr(v):
        h = v.astype(np.float32)
        if double:
            h = h.astype(np.float16).astype(np.float32)
        return np.array(mx.array(h).astype(mx.bfloat16).astype(mx.float32)
                        ).astype(np.float64)

    rows = []
    for mname, mod in pick:
        if id(mod) not in rec:
            print(f"  {mname}: not called in the forward, skipped")
            continue
        x, idx, srt = rec[id(mod)]
        IN, OUT = mod.input_dims, mod.output_dims
        idx_flat = idx.reshape(-1)
        N = idx_flat.size
        xf = np.broadcast_to(x, (*idx.shape, 1, IN)).reshape(N, IN).astype(np.float64)
        cb32 = np.array(mod.codebook.astype(mx.float32))
        D = cb32.shape[1]
        NSUB = IN // D
        sc32 = np.array(mod.vq_scales.astype(mx.float32))
        ref = np.zeros((N, OUT))
        s_x16 = np.zeros((N, OUT))
        s_w16 = np.zeros((N, OUT))
        x16 = f16(xf)
        for e in np.unique(idx_flat):
            rows_e = np.nonzero(idx_flat == e)[0]
            c = np.array(mod.codes[int(e)])
            if mod.pack_bits:
                c = vq_pack.unpack(c[None], NSUB, mod.pack_bits)[0]
            Wc = cb32[c.astype(np.int64)].reshape(OUT, IN)          # fp32 exact
            s_e = np.repeat(sc32[e].reshape(OUT, -1),
                            IN // sc32[e].reshape(OUT, -1).shape[1], axis=1)
            W = Wc.astype(np.float64) * s_e.astype(np.float64)
            W16 = (Wc * s_e).astype(np.float32).astype(np.float16).astype(np.float64)
            ref[rows_e] = xf[rows_e] @ W.T
            s_x16[rows_e] = x16[rows_e] @ W.T
            s_w16[rows_e] = x16[rows_e] @ W16.T
        rref = float(np.sqrt((ref ** 2).mean()))
        res = {}

        def err(v):
            return float(np.sqrt(((v - ref) ** 2).mean()))
        res["sim:out"] = err(outr(ref))
        res["sim:x16"] = err(outr(s_x16))
        res["sim:x16+w16"] = err(outr(s_w16))
        res["sim:out/f16"] = err(f16(ref))
        res["sim:x16/f16"] = err(f16(s_x16))
        res["sim:x16+w16/f16"] = err(f16(s_w16))
        kept = {}
        for an, kv in arms:
            saved = {}
            for k, v in kv.items():
                at = _attr(vqg, k)
                saved[at] = vqg[at]
                vqg[at] = v
            outs = {}
            try:
                # bf16 in -> bf16 out is the shipped I/O; its output rounding
                # (rel ~1.6e-3) swamps every kernel-internal term. The SAME
                # bf16 values fed as fp16 (exact: fp16 has more mantissa)
                # come back in fp16, 8x finer, so the kernel's own error
                # is visible ("arm/f16").
                for dt in (mx.bfloat16, mx.float16):
                    xm = mx.array(x).astype(dt)
                    y = mod(xm, mx.array(idx), sorted_indices=srt) if srt \
                        else mod(xm, mx.array(idx))
                    mx.eval(y)
                    outs[dt] = np.array(y.astype(mx.float32)).astype(
                        np.float64).reshape(N, OUT)
            finally:
                for at, v in saved.items():
                    vqg[at] = v
            res[an] = err(outs[mx.bfloat16])
            res[an + "/f16"] = err(outs[mx.float16])
            kept[an] = outs
        sub = float((np.abs(x16[x16 != 0]) < 6.103515625e-05).mean()) if (x16 != 0).any() else 0.0
        short = mname.split("model.")[-1]
        print(f"\n  {short}  N={N}  ref rms {rref:.4e}  |x|max {np.abs(xf).max():.3g}  "
              f"fp16-subnormal x {100*sub:.2f}%")
        for k, v in res.items():
            b = res["base/f16"] if k.endswith("/f16") else res["base"]
            print(f"    {k:22s} rms err {v:.4e}  rel {v/rref:.3e}  /base {v/b:.3f}")
        # correctness of the shipped bf16 output, element by element
        rb = outr(ref)
        first = arms[1][0] if len(arms) > 1 else None
        for an, _ in arms:
            yb, yh = kept[an][mx.bfloat16], kept[an][mx.float16]
            mis = float((yb != rb).mean())
            bias = float((yh - ref).mean()) / rref
            line = (f"    {an:22s} bf16 != round(exact) {100*mis:6.3f}%   "
                    f"mean signed err/f16 {bias:+.2e} (rel)")
            if first and an != first:
                line += f"   bf16 != {first} {100*float((yb != kept[first][mx.bfloat16]).mean()):6.3f}%"
            print(line)
            res[an + ":mis"] = mis
        rows.append((short, rref, res))
    if rows:
        print("\nSUMMARY  median over modules of (arm rms err / base rms err; "
              "':mis' = raw fraction of bf16 outputs not equal to the "
              "correctly rounded exact value)")
        for k in rows[0][2]:
            if k.endswith(":mis"):
                r = [res[k] for _, _, res in rows]
                print(f"    {k:22s} {100*float(np.median(r)):.3f}%   "
                      f"[{100*min(r):.3f} .. {100*max(r):.3f}]")
                continue
            b = "base/f16" if k.endswith("/f16") else "base"
            r = [res[k] / res[b] for _, _, res in rows]
            print(f"    {k:22s} {float(np.median(r)):.3f}   "
                  f"[{min(r):.3f} .. {max(r):.3f}]")
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
    attr = a.switch_flag if a.switch_flag in vqg else (
        "_" + a.switch_flag[3:] if a.switch_flag.startswith("VQ_") else a.switch_flag)
    if attr not in vqg:
        raise SystemExit(f"FAIL: runtime has no {attr} (wrong model.py?)")
    # OFF is the runtime's own value (False for a boolean flag; the shipped
    # number for a numeric knob such as VQ_FUSED_MAX_N), restored afterwards.
    off_val = vqg[attr]
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
            on_val = True if a.switch_value is None else _num(a.switch_value)
            for flag in (False, True):
                vqg[attr] = on_val if flag else off_val
                xm = mx.broadcast_to(mx.array(x).astype(mx.bfloat16)
                                     [None, :, None, None, :], (1, T, TOPK, 1, IN))
                y = mod(xm, mx.array(idx)[None])
                mx.eval(y)
                out[flag] = np.array(y.astype(mx.float32)).astype(
                    np.float64).reshape(T, TOPK, OUT)
            vqg[attr] = off_val
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
