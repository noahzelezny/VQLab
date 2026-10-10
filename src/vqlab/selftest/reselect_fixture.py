"""The reselect tool's self-check (was fit/reselect.py `selftest`, moved
here in untangle step 14): CPU only, no model.

    vqlab reselect selftest [--dir D]
    python -m vqlab.selftest.reselect_fixture [--dir D]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from vqlab import config
from vqlab.fit import reselect as R
from vqlab.records import provenance
from vqlab.runtime import vq_pack


def run(root=None):
    """Synthetic, CPU-only: a random 2-expert module, a codebook, a Gram
    with a planted anisotropic structure. Checks (1) re-selection never
    raises and on the whole lowers the training block-diagonal error vs
    nearest-neighbour (k-means) codes; (2) `apply` writes a NEW dir with a
    build record and leaves the base untouched. Returns the out dir."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(1234)
    root = root or os.path.join(config.scratch(), "reselect_selftest",
                                time.strftime("%Y%m%d-%H%M%S"))
    E, OUT, IN, D, K, group = 2, 16, 64, 4, 16, 32
    nsub = IN // D
    name = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    W = rng.standard_normal((E, OUT, IN)).astype(np.float32)
    Wn, scl = R.normalize(W, group)
    cb = rng.standard_normal((K, D)).astype(np.float32) * 0.5
    km = np.argmin(((Wn.reshape(-1, nsub, 1, D) - cb) ** 2).sum(-1), -1)   # k-means assign
    A = rng.standard_normal((IN, IN)).astype(np.float32)
    A[:, ::D] *= 8.0                                   # planted high-energy directions
    G = (A.T @ A / IN).astype(np.float32)
    blocks = R.gram_blocks(G, D)
    new = R.reselect_codes(Wn, cb, blocks)
    e_km = R.blockdiag_error(Wn, km, cb, blocks)
    e_new = R.blockdiag_error(Wn, new, cb, blocks)
    assert np.all(e_new <= e_km + 1e-5 * np.abs(e_km)), "re-selection raised error"
    assert e_new.sum() < 0.99 * e_km.sum(), (e_new.sum(), e_km.sum())

    art, tgt, bank = (os.path.join(root, x) for x in ("base", "teacher", "bank"))
    for d in (art, tgt, os.path.join(bank, "grams")):
        os.makedirs(d)
    bits = vq_pack.bits_for_k(K)
    vqm = {name: {"experts": E, "out": OUT, "in": IN, "k": K, "dim": D,
                  "group": group, "pack_bits": bits}}
    json.dump({"model_type": "synthetic", "vq_modules": vqm},
              open(os.path.join(art, "config.json"), "w"))
    json.dump({"model_type": "synthetic"}, open(os.path.join(tgt, "config.json"), "w"))
    mx.save_safetensors(os.path.join(art, "model.safetensors"), {
        name + ".codes": mx.array(vq_pack.pack(km.reshape(E, OUT, nsub), bits)),
        name + ".codebook": mx.array(cb).astype(mx.float16),
        name + ".vq_scales": mx.array(scl).astype(mx.float16)})
    cb = np.array(mx.array(cb).astype(mx.float16).astype(mx.float32))  # as stored
    mx.save_safetensors(os.path.join(tgt, "model.safetensors"),
                        {name + ".weight": mx.array(W)})
    mx.save_safetensors(os.path.join(bank, "grams", R._fname(name)), {"gram": mx.array(G)})
    json.dump({"schema": R.SCHEMA, "modules": [name], "seed": 1234, "tokens": 0,
               "model": "synthetic"}, open(os.path.join(bank, R.MANIFEST), "w"))
    base_before = R._sha256(os.path.join(art, "model.safetensors"))
    out = os.path.join(root, "out")
    rc = R.main(["apply", "--artifact", art, "--grams", bank, "--target", tgt,
               "--out", out, "--device", "cpu"])
    assert rc == 0
    assert R._sha256(os.path.join(art, "model.safetensors")) == base_before, "base edited"
    rec = provenance.load(out)
    assert rec["tool"]["name"] == "reselect"
    assert rec["method"]["calibration_manifest_sha256"] == R._sha256(os.path.join(bank, R.MANIFEST))
    r = rec["modules"][name]
    assert r["J_train_new"] <= r["J_train_base_codes"] and r["J_train_gain"] > 0.01, r
    got = vq_pack.unpack(np.array(mx.load(os.path.join(out, "model.safetensors"))[name + ".codes"]),
                         nsub, bits).reshape(-1, nsub)
    assert np.array_equal(got, R.reselect_codes(Wn, cb, blocks)), "stored codes != re-selected"
    print(f"SELFTEST OK: J_train {e_km.sum():.3f} -> {e_new.sum():.3f} "
          f"({100 * (1 - e_new.sum() / e_km.sum()):.1f}% lower); apply gain "
          f"{100 * r['J_train_gain']:.1f}%; new dir {out}")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["selftest"]:          # routed from `vqlab reselect selftest`
        argv = argv[1:]
    ap = argparse.ArgumentParser(prog="vqlab reselect selftest",
                                 description="CPU synthetic check (no model, no GPU)")
    ap.add_argument("--dir", default=None)
    run(ap.parse_args(argv).dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
