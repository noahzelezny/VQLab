"""Selftest: the packer (a raw fit's codes, bit-packed)."""
from __future__ import annotations

import json

import mlx.core as mx

from vqlab._layout import find as _find
from vqlab.selftest import check, run
from vqlab.selftest.synthetic_model import decode_all


def check_packer(ctx):
    print("[3/7] packer")
    f1, A = ctx.f1, ctx.A
    ctx.packed = packed = ctx.tmp / "packed"
    # `pack` (expert format, [E, OUT, NSUB]) is what packs a raw fit's
    # vq_modules; pack-dense takes a BUILT dense artifact's vq_linear.
    # Until 2026-10-03 this step ran pack-dense on the raw fit, which
    # packed nothing: "smaller" passed on header bytes and "bit-exact"
    # compared the unpacked codes with themselves.
    p = run([str(_find("pack_artifact.py")), "--src", str(f1), "--out", str(packed)],
            verbose=ctx.verbose)
    if check("pack runs", p.returncode == 0):
        u = sum(f.stat().st_size for f in f1.glob("*.safetensors"))
        q = sum(f.stat().st_size for f in packed.glob("*.safetensors"))
        pk = json.load(open(packed / "config.json"))["vq_modules"]
        check("every module packed to 4 bits (K16)",
              all(m.get("pack_bits") == 4 for m in pk.values()),
              f"{sum(bool(m.get('pack_bits')) for m in pk.values())}/{len(pk)}")
        check("packed artifact is smaller than unpacked", q < u * 0.9,
              f"{q} < {u} bytes")
        D_ = decode_all(packed)
        worst = max(float(mx.max(mx.abs(A[k] - D_[k])).item()) for k in A)
        check("packing is bit-exact (decode unchanged)", worst == 0.0,
              f"max|delta| {worst}")
