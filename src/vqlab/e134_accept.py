"""E134 acceptance: device-memory d4 codebook kernels.

Bar (set with the paper + M3 sessions BEFORE the fix existed):
  1. bit-identical to the VERIFIED threadgroup kernels wherever both load
  2. correct vs a reference dequant-matmul where only devcb loads
  3. K256/K2048 unchanged; K4096/K8192 move FAIL -> works
Run at multiple shapes, unpacked and packed.
"""
import sys
import mlx.core as mx
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import vq_switch, vq_pack

mx.random.seed(0)


def reference(x, eidx, codes, codebook, scales, D, G):
    """Ground truth: reconstruct W from codes and matmul in float32."""
    E, OUT, NSUB = codes.shape
    IN = NSUB * D
    outs = []
    for t in range(x.shape[0]):
        e = int(eidx[t])
        cb = codebook.astype(mx.float32)
        w = cb[codes[e].reshape(-1).astype(mx.int32)].reshape(OUT, NSUB, D)
        w = w.reshape(OUT, IN // G, G) * scales[e].astype(mx.float32)[:, :, None]
        outs.append((x[t].astype(mx.float32)[None, :] @ w.reshape(OUT, IN).T)[0])
    return mx.stack(outs)


def make(E, OUT, IN, K, G=64, D=4):
    NSUB = IN // D
    codes = mx.random.randint(0, K, (E, OUT, NSUB)).astype(
        mx.uint8 if K <= 256 else mx.uint16)
    codebook = (mx.random.normal((K, D)) * 0.1).astype(mx.float16)
    scales = (mx.random.uniform(shape=(E, OUT, IN // G)) * 0.5 + 0.5).astype(mx.float16)
    return codes, codebook, scales


SHAPES = [(4, 512, 2048), (2, 256, 1024)]
KS = [256, 2048, 4096, 8192]
fails = 0
print("%-22s %-6s %-8s %-10s %s" % ("shape", "K", "path", "vs-ref", "vs-threadgroup"))
for (E, OUT, IN) in SHAPES:
    for K in KS:
        NSUB = IN // 4
        codes, cb, sc = make(E, OUT, IN, K)
        N = 8
        x = (mx.random.normal((N, IN)) * 0.5).astype(mx.float16)
        eidx = mx.random.randint(0, E, (N,)).astype(mx.uint32)
        ref = reference(x, eidx, codes, cb, sc, 4, 64)
        fits = vq_switch._d4_tg_fits(K, NSUB)

        # --- devcb path (force it on, so it is exercised at EVERY K) ---
        saved = vq_switch._TG_CAP_BYTES
        vq_switch._TG_CAP_BYTES = 0            # force device-memory
        y_dev = vq_switch._fused(x, eidx, codes, cb, sc)
        mx.eval(y_dev)
        vq_switch._TG_CAP_BYTES = saved

        rel = float(mx.max(mx.abs(y_dev.astype(mx.float32) - ref))
                    / mx.maximum(mx.max(mx.abs(ref)), 1e-6))

        # --- threadgroup path, where it loads ---
        tg_note = "n/a (over cap)"
        if fits:
            vq_switch._TG_CAP_BYTES = 1 << 30  # force threadgroup
            y_tg = vq_switch._fused(x, eidx, codes, cb, sc)
            mx.eval(y_tg)
            vq_switch._TG_CAP_BYTES = saved
            same = bool(mx.array_equal(y_dev, y_tg))
            tg_note = "BIT-IDENTICAL" if same else "*** DIFFERS ***"
            if not same:
                fails += 1
        ok_ref = rel < 2e-2
        if not ok_ref:
            fails += 1
        print("%-22s %-6d %-8s %-10s %s" % (
            f"E{E} OUT{OUT} IN{IN}", K, "devcb",
            ("ok %.1e" % rel) if ok_ref else ("BAD %.1e" % rel), tg_note))

# --- packed variants ---
print()
print("PACKED:")
for (E, OUT, IN) in SHAPES:
    for K in KS:
        bits = int(K - 1).bit_length()
        NSUB = IN // 4
        if NSUB % 32:
            continue
        codes, cb, sc = make(E, OUT, IN, K)
        packed = mx.array(vq_pack.pack(__import__("numpy").array(codes), bits))
        N = 8
        x = (mx.random.normal((N, IN)) * 0.5).astype(mx.float16)
        eidx = mx.random.randint(0, E, (N,)).astype(mx.uint32)
        ref = reference(x, eidx, codes, cb, sc, 4, 64)
        saved = vq_switch._TG_CAP_BYTES
        fits = vq_switch._d4_tg_fits(K, NSUB)   # BEFORE the override, or it
                                                # always reads False and the
                                                # bit-identity check silently
                                                # never runs
        vq_switch._TG_CAP_BYTES = 0
        y_dev = vq_switch._fused(x, eidx, packed, cb, sc, pack_bits=bits)
        mx.eval(y_dev)
        tg_note = "n/a (over cap)"
        if fits:
            vq_switch._TG_CAP_BYTES = 1 << 30
            y_tg = vq_switch._fused(x, eidx, packed, cb, sc, pack_bits=bits)
            mx.eval(y_tg)
            same = bool(mx.array_equal(y_dev, y_tg))
            tg_note = "BIT-IDENTICAL" if same else "*** DIFFERS ***"
            if not same:
                fails += 1
        vq_switch._TG_CAP_BYTES = saved
        rel = float(mx.max(mx.abs(y_dev.astype(mx.float32) - ref))
                    / mx.maximum(mx.max(mx.abs(ref)), 1e-6))
        ok_ref = rel < 2e-2
        if not ok_ref:
            fails += 1
        print("%-22s %-6d bits=%-3d %-10s %s" % (
            f"E{E} OUT{OUT} IN{IN}", K, bits,
            ("ok %.1e" % rel) if ok_ref else ("BAD %.1e" % rel), tg_note))

print()
print("ACCEPTANCE:", "PASS" if fails == 0 else f"FAIL ({fails} checks)")
sys.exit(1 if fails else 0)
