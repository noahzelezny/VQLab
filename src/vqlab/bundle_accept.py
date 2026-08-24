"""E134 acceptance re-run against the BUNDLED copy — the code a downloader runs.

Gap this closes: the original acceptance imported the repo checkout's
vq_switch.py, i.e. the copy the developers bench. On a patched box the load_model hook
re-imports VQSwitchLinear from site-packages after the bundle has built the
model, so our benches exercise site-packages for MoE artifacts while users
execute the bundle. Same text is not the same test.

Method: lift the vq_switch half out of the artifact's model.py (everything
before the loader shim), import THAT as the module under test, and run the
same bit-identity + reference checks.
"""
import sys, pathlib, importlib.util
import mlx.core as mx

import tempfile
if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    print("usage: vqlab bundle-accept <artifact-dir>\n\n"
          "Lifts the runtime out of <artifact-dir>/model.py and runs the\n"
          "kernel acceptance checks against THAT copy — the code a\n"
          "downloader executes — rather than whatever is on the import path.")
    sys.exit(0)
if len(sys.argv) != 2:
    sys.exit("usage: vqlab bundle-accept <artifact-dir>")
ART = sys.argv[1]
bundle = pathlib.Path(ART) / "model.py"
text = bundle.read_text()
cut = text.find("import importlib as _importlib")
assert cut > 0, "shim marker not found"
tmp = pathlib.Path(tempfile.mkdtemp(prefix="vqlab_bundle_")) / "_bundled_vq_switch.py"
tmp.write_text(text[:cut])

spec = importlib.util.spec_from_file_location("bundled_vq_switch", tmp)
bvq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bvq)
print("module under test:", bvq.__file__)
print("has devcb kernels:",
      hasattr(bvq, "_SRC_FUSED_D4_DEVCB"), hasattr(bvq, "_SRC_FUSED_PACKED_D4_DEVCB"))
print("has cap guard:", hasattr(bvq, "_d4_tg_fits"))

mx.random.seed(0)

def reference(x, eidx, codes, codebook, scales, D, G):
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

fails = 0
for (E, OUT, IN) in [(4, 512, 2048), (2, 256, 1024)]:
    for K in [256, 2048, 4096, 8192]:
        NSUB = IN // 4
        codes = mx.random.randint(0, K, (E, OUT, NSUB)).astype(
            mx.uint8 if K <= 256 else mx.uint16)
        cb = (mx.random.normal((K, 4)) * 0.1).astype(mx.float16)
        sc = (mx.random.uniform(shape=(E, OUT, IN // 64)) * 0.5 + 0.5).astype(mx.float16)
        x = (mx.random.normal((8, IN)) * 0.5).astype(mx.float16)
        eidx = mx.random.randint(0, E, (8,)).astype(mx.uint32)
        ref = reference(x, eidx, codes, cb, sc, 4, 64)
        fits = bvq._d4_tg_fits(K, NSUB)
        saved = bvq._TG_CAP_BYTES
        bvq._TG_CAP_BYTES = 0
        y_dev = bvq._fused(x, eidx, codes, cb, sc); mx.eval(y_dev)
        note = "n/a (over cap)"
        if fits:
            bvq._TG_CAP_BYTES = 1 << 30
            y_tg = bvq._fused(x, eidx, codes, cb, sc); mx.eval(y_tg)
            same = bool(mx.array_equal(y_dev, y_tg))
            note = "BIT-IDENTICAL" if same else "*** DIFFERS ***"
            if not same: fails += 1
        bvq._TG_CAP_BYTES = saved
        rel = float(mx.max(mx.abs(y_dev.astype(mx.float32) - ref))
                    / mx.maximum(mx.max(mx.abs(ref)), 1e-6))
        if rel >= 2e-2: fails += 1
        print("  E%d OUT%-4d IN%-5d K%-6d ref %.1e  %s" % (E, OUT, IN, K, rel, note))

print("BUNDLED-COPY ACCEPTANCE:", "PASS" if fails == 0 else "FAIL (%d)" % fails)
sys.exit(1 if fails else 0)
