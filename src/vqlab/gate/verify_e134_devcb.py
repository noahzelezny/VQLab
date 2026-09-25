"""E113 acceptance step 1 for the E134 device-codebook d4 kernel.

Compares _SRC_FUSED_PACKED (threadgroup cb) against _SRC_FUSED_PACKED_D4_DEVCB
(device cb) THROUGH THE REAL _fused DISPATCH — the only difference forced is
where cb is read from, by monkeypatching _d4_tg_fits. Both kernels must load,
so this runs at K<=2048 where the threadgroup variant still fits Metal's cap.
At K=8192 only devcb exists, but the kernel source is K-agnostic, so agreement
here is what licenses it there.

III.10 corollary: instrument the REAL code. We do not reimplement the call.
"""
import json, os, sys, struct
import mlx.core as mx
import mlx_lm.models.vq_switch as vs

import os
E = os.environ.get("VQLAB_MODELS_DIR") or (sys.argv[1] if len(sys.argv) > 1 else ".")

def load_layer(art):
    idx = json.load(open(f"{art}/model.safetensors.index.json"))["weight_map"]
    pref = None
    for k in idx:
        if k.endswith(".codes") and "switch_mlp" in k:
            pref = k[:-len(".codes")]; break
    if pref is None: return None
    shards = {idx[f"{pref}.{s}"] for s in ("codes","codebook","vq_scales")}
    t = {}
    for sh in shards:
        t.update(mx.load(os.path.join(art, sh)))
    return (t[f"{pref}.codes"], t[f"{pref}.codebook"], t[f"{pref}.vq_scales"], pref)

def pack_bits_of(art):
    c = json.load(open(f"{art}/config.json"))
    return _pb(c)

def _pb(c):
    s = json.dumps(c)
    if '"pack_bits"' not in s: return 0
    return int(s.split('"pack_bits":')[1].strip().split(",")[0].split("}")[0])

ARTS = [f"{E}/qwen36-35b-rungs/vq-K2048-d4-packed", f"{E}/qwen36-35b-rungs/vq-K256-d4"]
fails = 0; ran = 0
for art in ARTS:
    if not os.path.isdir(art):
        print(f"SKIP (missing) {os.path.basename(art)}"); continue
    got = load_layer(art)
    if got is None:
        print(f"SKIP (no switch_mlp) {os.path.basename(art)}"); continue
    codes, cb, sc, pref = got
    pb = _pb(json.load(open(f"{art}/config.json")))
    K, D = int(cb.shape[0]), int(cb.shape[1])
    words = int(codes.shape[-1])
    NSUB = (words * 32) // pb if pb else words
    print(f"\n=== {os.path.basename(art)}  K={K} D={D} pack_bits={pb} "
          f"tg_fits={vs._d4_tg_fits(K, NSUB)}")
    if D != 4:
        print("   not d4, skipping"); continue
    E_experts = int(codes.shape[0])
    IN = int(cb.shape[1]) * NSUB
    for N in (1, 8, 32):
        mx.random.seed(0)
        x = mx.random.normal((N, IN)).astype(mx.float16)
        eidx = mx.zeros((N,), dtype=mx.uint32)
        orig = vs._d4_tg_fits
        seen = []
        _gk = vs._get_kernel
        vs._get_kernel = lambda name, src, _g=_gk, _s=seen: (_s.append(name) or _g(name, src))
        try:
            vs._d4_tg_fits = lambda k, n: True          # force THREADGROUP
            y_tg = vs._fused(x, eidx, codes, cb, sc, pack_bits=pb); mx.eval(y_tg)
            vs._d4_tg_fits = lambda k, n: False         # force DEVICE cb
            y_dev = vs._fused(x, eidx, codes, cb, sc, pack_bits=pb); mx.eval(y_dev)
        except Exception as ex:
            print(f"   N={N:<4} ERROR {type(ex).__name__}: {str(ex)[:110]}")
            vs._d4_tg_fits = orig; fails += 1; continue
        vs._d4_tg_fits = orig
        vs._get_kernel = _gk
        k_tg, k_dev = (seen[0], seen[-1]) if len(seen) >= 2 else (seen + ["?", "?"])[:2]
        fired = k_tg != k_dev
        same = bool(mx.array_equal(y_tg, y_dev))
        maxdiff = float(mx.max(mx.abs(y_tg.astype(mx.float32) - y_dev.astype(mx.float32))))
        ran += 1
        if not same: fails += 1
        if not fired:
            print(f"   N={N:<4} VACUOUS — same kernel both runs ({k_tg}); probe did not fire")
            fails += 1; ran -= 1
        else:
            print(f"   N={N:<4} bit-identical={same}  max|diff|={maxdiff:.3e}")
            print(f"          tg={k_tg}")
            print(f"          dev={k_dev}")
print(f"\nRESULT: {ran} VALID comparisons, {fails} failures/vacuous")
sys.exit(1 if fails or ran == 0 else 0)
