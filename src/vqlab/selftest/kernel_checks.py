"""Selftest: the runtime kernels, bundled and installed."""
from __future__ import annotations

import mlx.core as mx

from vqlab._layout import find as _find
from vqlab.selftest import check
from vqlab.selftest.synthetic_model import G, IN_D, OUT_D


def check_runtime_kernels(ctx):
    print("[6/7] runtime kernels")
    bundle_txt = ((_find("vq_switch.py")).read_text()
                  + (_find("vq_dense.py")).read_text())
    ns = {"__name__": "bundled_model"}
    exec(compile(bundle_txt, "model.py", "exec"), ns)
    K_, D_d = 16, 2
    codes = mx.random.randint(0, K_, (OUT_D, IN_D // D_d)).astype(mx.uint8)
    cb = (mx.random.normal((K_, D_d)) * 0.1).astype(mx.float16)
    sc = (mx.random.uniform(shape=(OUT_D, IN_D // G)) * .5 + .5).astype(mx.float16)
    x = (mx.random.normal((1, IN_D)) * 0.5).astype(mx.float16)
    yb = ns["VQLinear"](codes, cb, sc, group_size=G, pack_bits=0)(x)
    mx.eval(yb)
    from vqlab.runtime import vq_dense
    yi = vq_dense.VQLinear(codes, cb, sc, group_size=G, pack_bits=0)(x)
    mx.eval(yi)
    check("bundled and installed runtimes agree bit-for-bit",
          bool(mx.array_equal(yb, yi)))

    # Kernel RESOLUTION: name which copy each path actually uses
    # (III.13 — never assume). The bundle must resolve from its own
    # globals; the standalone module must resolve to the package sibling,
    # NOT to mlx_lm.models.vq_switch, which exists only on VQ-patched
    # lab machines. The original version of this check asserted the
    # standalone path FAILS without mlx_lm — true of the old fallback
    # chain, fixed the first time the selftest ran in a fresh venv.
    bundled_fn = ns.get("_dense_fused")
    check("bundle resolves _dense_fused from its own globals",
          bundled_fn is not None)
    resolved = vq_dense._resolve_kernel("_dense_fused")
    rmod = getattr(resolved, "__module__", "?")
    check("standalone resolves the sibling, not a patched mlx_lm",
          "mlx_lm" not in rmod, f"resolved from {rmod}")
