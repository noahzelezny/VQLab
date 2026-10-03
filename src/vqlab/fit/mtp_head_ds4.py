#!/usr/bin/env python3
"""mtp-head-ds4: build DeepSeek-V4-Flash's draft head for a VQ trunk, in one
command (devlist #9; promoted from three scratch scripts of 2026-10-02).

    vqlab mtp-head-ds4 --release <official release dir, F8_E8M0 relabelled>
                       --out mtp-head-q6.safetensors
                       [--experts vq|exact] [--k 2048 --dim 4] [--dense-bits 6|8|0]

Three steps, each the recipe the shipped head was built with:
1. EXACT: the release's mtp.0.* tensors through the runtime's own
   deepseek_v4 sanitize (presented as a temporary layer 0, renamed back):
   routed experts stay mxfp4 bit-identical, FP8 dequantizes exactly to bf16.
2. --experts vq (default): the three routed-expert projections refit at the
   trunk's geometry (d4/K2048, seed 1234) from the release's FP4 experts,
   packed; vq_modules stored in the file's metadata.
3. --dense-bits 6 (default; 0 keeps bf16): attention, e_proj/h_proj and the
   shared expert affine-quantized, group 64. Router, norms and
   hyper-connections stay exact.
Measured 2026-10-02 (F-log, A/B through Knurlogic): VQ, affine-8 and affine-6
heads have identical acceptance (0.885 tensor split, 0.95 one Mac); the
6-bit head shipped at 2.38 GiB.

Not for Vision-Exp: its DSpark drafter (3 mtp layers, Markov + confidence
heads) is a different module.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np

from vqlab import _layout  # noqa: F401
from vqlab.core import expert_src as ES

DENSE = (".attn.wq", ".attn.wo", "e_proj", "h_proj", "shared_experts")


def exact_head(release: pathlib.Path) -> dict:
    cfg = json.load(open(release / "config.json"))
    if cfg.get("num_nextn_predict_layers", 0) != 1 or cfg.get("dspark_block_size"):
        raise SystemExit("this release does not carry ONE classic MTP layer (num_nextn_predict_layers=1, "
                         "no DSpark); mtp-head-ds4 builds DeepSeek-V4-Flash's head only")
    try:
        from mlx_lm.utils import _get_classes
        mc, ac = _get_classes(cfg)
    except Exception:                             # model_type unknown to this mlx-lm
        from knurlogic.engine import register as kreg
        kreg.register("deepseek_v4")
        from mlx_lm.utils import _get_classes
        mc, ac = _get_classes(cfg)
    model = mc(ac.from_dict(cfg))                 # never evaluated
    wm = json.load(open(release / "model.safetensors.index.json"))["weight_map"]
    keys = [k for k in wm if k.startswith("mtp.0.")]
    if not keys:
        raise SystemExit(f"{release}: no mtp.0.* tensors")
    with mx.stream(mx.cpu):
        sub = {}
        for sh in sorted({wm[k] for k in keys}):
            d = mx.load(str(release / sh))
            sub.update({k.replace("mtp.0.", "layers.0.", 1): d[k] for k in keys if wm[k] == sh})
        res = model.sanitize(sub)
        out = {k.replace("model.layers.0.", "mtp.0.", 1): v for k, v in res.items()}
        mx.eval(list(out.values()))
    bad = [k for k in out if not k.startswith("mtp.0.")]
    if bad:
        raise SystemExit(f"sanitize produced non-head tensors, e.g. {bad[:3]}")
    print(f"exact head: {len(keys)} -> {len(out)} tensors", flush=True)
    return out


def vq_experts(head, release, D, K):
    from vqlab.fit import geo_build as GB
    from vqlab.runtime import vq_pack
    fam = {"src_key": "mtp.0.ffn.experts.{e}.{key}.weight", "src_quant": "mxfp4",
           "proj": {"gate_proj": ("w1", None), "up_proj": ("w3", None), "down_proj": ("w2", None)}}
    idx = json.load(open(release / "model.safetensors.index.json"))["weight_map"]
    rng = np.random.default_rng(1234)
    vq = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        t0 = time.time()
        W = ES.load_expert_stack(str(release), idx, fam, 0, proj).astype(mx.float32)
        cb, codes, scales = GB.fit_module(W, D, K, rng)
        p = f"mtp.0.ffn.switch_mlp.{proj}"
        for s in (".weight", ".scales"):
            head.pop(p + s, None)
        head[p + ".codes"], head[p + ".codebook"], head[p + ".vq_scales"] = codes, cb, scales
        E, O, I = W.shape
        vq[p] = {"experts": E, "out": O, "in": I, "dim": D, "k": K, "group": 64,
                 "pack_bits": vq_pack.bits_for_k(K)}
        print(f"  {proj}: {list(W.shape)} -> d{D}/K{K} ({time.time() - t0:.0f}s)", flush=True)
        del W
    return vq


def quantize_dense(head, bits):
    dense = [k for k in head if k.endswith(".weight") and head[k].ndim == 2
             and head[k].dtype == mx.bfloat16 and any(s in k for s in DENSE)]
    for k in dense:
        w, s, b = mx.quantize(head[k], group_size=64, bits=bits)
        p = k[:-len(".weight")]
        head[p + ".weight"], head[p + ".scales"], head[p + ".biases"] = w, s, b
    print(f"dense: {len(dense)} projections -> affine {bits}-bit group 64", flush=True)
    return len(dense)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab mtp-head-ds4", description=__doc__.split("\n")[0])
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", choices=("vq", "exact"), default="vq")
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--dense-bits", type=int, default=6, help="0 keeps the dense projections bf16")
    a = ap.parse_args(argv)
    release, out = pathlib.Path(a.release), pathlib.Path(a.out)
    head = exact_head(release)
    meta = {"format": "mlx", "head": "deepseek_v4 mtp.0 (num_nextn_predict_layers=1)",
            "built_by": "vqlab mtp-head-ds4", "seed": "1234"}
    prec = ["FP8 -> bf16 exact dequant"]
    if a.experts == "vq":
        vq = vq_experts(head, release, a.dim, a.k)
        meta["vq_modules"] = json.dumps(vq)
        prec.append(f"routed experts VQ d{a.dim}/K{a.k} packed (trunk geometry)")
    else:
        prec.append("routed experts mxfp4 bit-identical to the release")
    if a.dense_bits:
        quantize_dense(head, a.dense_bits)
        prec.append(f"dense projections affine {a.dense_bits}-bit group 64")
    meta["precision"] = "; ".join(prec)
    mx.save_safetensors(str(out), head, metadata=meta)
    print(f"{out.name}: {len(head)} tensors, {os.path.getsize(out) / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
