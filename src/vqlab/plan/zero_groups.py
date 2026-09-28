"""zero-groups: how many code bytes encode NOTHING -- sized from scales alone, no GPU.

Why. `vqlab fits census` (fitstore.code_usage) found the 397B teacher's
early-layer experts are mostly near-zero rows: 86% of the 64-element weight
groups at L0 have max|w| ~1e-29 (L1 81%, L10 35%, L30 5%; the 35B's L0 41%).
Those groups get vq_scales == 0 -- fp16 underflow, the CORRECT encoding --
but every shipped format still stores a full set of codes for them. This
prices a "skip zero groups" format BEFORE anyone writes a kernel for it
(FINDINGS I.5: price a rung before building it): the GiB it would save, per
layer and as a share of the artifact's text bytes.

"Zero-scale" is counted two ways. `exact` is scale == 0. `tiny` is
|scale| < 6.2e-5 (below the fp16 normal minimum), because some older fitters
stored a small floor instead of 0; tiny includes exact.

Reads safetensors HEADERS plus the `*.vq_scales` tensors only (a few MB per
module); never the codes, never mlx. Safe to run while another session holds
the GPU.

    vqlab zero-groups <artifact_dir> [...] [--json out.json]
    vqlab zero-groups --fits [--family F] [--teacher T] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import fitstore  # noqa: E402
import registry  # noqa: E402

TINY = 6.2e-5          # fp16 smallest normal is 6.10e-5
GIB = 2 ** 30
_NP = {"F16": np.float16, "BF16": None, "F32": np.float32}


def _scales(path, h, base, key):
    t = h[key]
    raw = fitstore.read_tensor_bytes(path, h, base, key)
    if t["dtype"] == "BF16":
        a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = np.frombuffer(raw, _NP[t["dtype"]])
    return np.abs(a.astype(np.float32))


def count_module(path, h, base, key, d, K, pack_bits=None, group=fitstore.GSZ):
    """Zero-group counts and the code bytes they occupy for one module.
    Works for MoE [E, OUT, NGRP] and dense [OUT, NGRP] alike: a group is one
    scale, and costs (group/d) codes of pack_bits (else ceil(log2 K)) bits."""
    a = _scales(path, h, base, key)
    bits = pack_bits or math.ceil(math.log2(K))
    per = (group // d) * bits / 8
    exact, tiny = int((a == 0).sum()), int((a < TINY).sum())
    return {"groups": int(a.size), "zero": exact, "tiny": tiny,
            "code_bytes": a.size * per, "zero_bytes": exact * per, "tiny_bytes": tiny * per}


def _layer(name):
    m = fitstore._MOD_RX.search(name)
    return int(m.group(1)) if m else -1


def _add(acc, r):
    for k, v in r.items():
        acc[k] = acc.get(k, 0) + v


def scan_artifact(d):
    d = pathlib.Path(d)
    cfg = json.loads((d / "config.json").read_text())
    mods = registry.vq_specs(cfg)          # incl. vq_ple (its own group size)
    geo, _ = registry.geometry_mix(cfg)
    shards = sorted(glob.glob(str(d / "*.safetensors")))
    text = sum(os.path.getsize(os.path.realpath(f)) for f in shards)
    total, layers = {}, collections.defaultdict(dict)
    seen = 0
    for f in shards:
        h, base = fitstore.read_header(f)
        for key in h:
            if not key.endswith(".vq_scales"):
                continue
            m = key[: -len(".vq_scales")]
            c = mods.get(m)
            if c is None:  # no config entry: take geometry from the header
                K, dd = h[m + ".codebook"]["shape"][-2:]
                c = {"k": K, "dim": dd, "pack_bits": None, "group": fitstore.GSZ}
            r = count_module(f, h, base, key, int(c["dim"]), int(c["k"]),
                             c.get("pack_bits"), int(c.get("group") or fitstore.GSZ))
            _add(total, r)
            _add(layers[_layer(m)], r)
            seen += 1
    return {"artifact": d.name, "path": str(d), "geometry": geo,
            "modules_config": len(mods), "modules_read": seen,
            "text_bytes": text, "total": total, "layers": dict(sorted(layers.items()))}


def scan_fits(family=None, teacher=None):
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for root in fitstore.roots():
        for rec in fitstore.read_index(root).values():
            if family and rec.get("family") != family or teacher and rec.get("teacher") != teacher:
                continue
            p = rec["location"].split("#", 1)[0]
            if not os.path.exists(p):
                continue
            h, base = fitstore.read_header(p)
            key = rec["module"] + ".vq_scales"
            if key not in h:
                continue
            r = count_module(p, h, base, key, rec["d"], rec["K"], rec.get("pack_bits"))
            r["fits"] = 1
            _add(out[f"{rec.get('family')}/{rec.get('teacher')}"][rec.get("layer", -1)], r)
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def print_artifact(r, top=8):
    t = r["total"]
    print(f"\n{r['artifact']}  ({r['modules_read']}/{r['modules_config']} VQ modules, "
          f"text {r['text_bytes'] / GIB:.2f} GiB)")
    if not t:
        print("  no vq_scales found")
        return
    print(f"  zero groups   exact {_pct(t['zero'], t['groups']):6.2f}%   "
          f"tiny {_pct(t['tiny'], t['groups']):6.2f}%")
    print(f"  saveable      exact {t['zero_bytes'] / GIB:7.3f} GiB "
          f"({_pct(t['zero_bytes'], r['text_bytes']):.2f}% of text)   "
          f"tiny {t['tiny_bytes'] / GIB:7.3f} GiB ({_pct(t['tiny_bytes'], r['text_bytes']):.2f}%)")
    rows = sorted(r["layers"].items(), key=lambda kv: -kv[1]["tiny_bytes"])[:top]
    print("  top layers    layer  exact%   tiny%   tiny GiB")
    for L, v in rows:
        print(f"               {L:5d}  {_pct(v['zero'], v['groups']):6.2f}  "
              f"{_pct(v['tiny'], v['groups']):6.2f}  {v['tiny_bytes'] / GIB:8.3f}")


def print_fits(res):
    for k, layers in res.items():
        g = {}
        for v in layers.values():
            _add(g, v)
        print(f"\n{k}: {g['fits']} fits, exact {_pct(g['zero'], g['groups']):.2f}%  "
              f"tiny {_pct(g['tiny'], g['groups']):.2f}%  tiny {g['tiny_bytes'] / GIB:.3f} GiB "
              f"of {g['code_bytes'] / GIB:.3f} GiB codes")
        for L, v in layers.items():
            print(f"  L{L:<3d} fits {v['fits']:4d}  exact {_pct(v['zero'], v['groups']):6.2f}%  "
                  f"tiny {_pct(v['tiny'], v['groups']):6.2f}%  {v['tiny_bytes'] / GIB:.3f} GiB")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab zero-groups", description=__doc__.split("\n\n")[0])
    ap.add_argument("artifacts", nargs="*")
    ap.add_argument("--fits", action="store_true", help="size from the fit store index instead")
    ap.add_argument("--family")
    ap.add_argument("--teacher")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", help="write results here (storage array scratch, not the internal disk)")
    a = ap.parse_args(argv)
    if a.fits:
        res = scan_fits(a.family, a.teacher)
        print_fits(res)
    else:
        if not a.artifacts:
            ap.error("give artifact dirs or --fits")
        res = []
        for d in a.artifacts:
            r = scan_artifact(d)
            print_artifact(r, a.top)
            res.append(r)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
