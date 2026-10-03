#!/usr/bin/env python3
"""mix: assemble an artifact from per-layer-band sources, shard by shard.

    vqlab mix --out OUT --base SRC [--band SRC:LO-HI ...] [--copy]

Every shard of --base is taken from --base unless a later --band covers ALL
the layers that shard holds (later bands win). A shard holding layers both
inside and outside a band is refused: the band must be widened or narrowed to
shard boundaries (`--plan` prints each shard's layers). Config maps follow
the bytes: for every module whose tensors come from a source, that source's
`vq_modules`, `quantization` and `vq_skipzero` entries are taken and the
others dropped. Everything that is not a weight shard (model.py, tokenizer,
template, card) comes from --base; a source whose model.py differs is refused
(two runtimes in one artifact is a runtime nobody scored).

Default links shards (symlinks: a scoring candidate costs no disk); --copy
makes real files (APFS clones where the volume supports them) for a release.
This is how DeepSeek-V4-Flash-VQ-3.2 (K2048 + K4096 on L24-35) was built.
"""
from __future__ import annotations

import argparse
import collections
import filecmp
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

SKIP = ("config.json", "model.safetensors.index.json", "vqlab_provenance.json",
        "vqlab_provenance.history.jsonl")


def _idx(src):
    return json.load(open(src / "model.safetensors.index.json"))["weight_map"]


def shard_layers(wmap, vq):
    """shard -> the layers of the VQ modules it holds. Only VQ modules count:
    the band decides which CODES a layer gets; the rest of a shard (norms,
    attention of a neighbouring layer) is skeleton, identical in every source
    of one mix (checked by header in build())."""
    out = collections.defaultdict(set)
    for k, f in wmap.items():
        out[f]
        if _module(k) not in vq:
            continue
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", k)
        out[f].add(int(m.group(1)) if m else -1)
    return out


def _vq(src):
    return set(json.load(open(src / "config.json")).get("vq_modules", {}))


def plan(base, bands):
    wmap = _idx(base)
    vq = _vq(base).union(*[_vq(s) for s, _, _ in bands])
    layers = shard_layers(wmap, vq)
    pick = {f: base for f in layers}
    for src, lo, hi in bands:
        want = set(range(lo, hi + 1))
        # a band source may be partial (a fit of only those layers), but every
        # shard the band selects must exist there holding the SAME layers
        slayers = shard_layers(_idx(src), vq)
        for f, ls in layers.items():
            if ls & want:
                if not ls <= want:
                    raise SystemExit(f"{f} holds layers {sorted(ls)}: straddles band "
                                     f"{lo}-{hi}; align the band to shard boundaries")
                if slayers.get(f) != ls:
                    raise SystemExit(f"{src}: shard {f} is missing or holds other layers "
                                     f"({sorted(slayers.get(f, []))} vs {sorted(ls)}); "
                                     "mix needs the same skeleton")
                pick[f] = src
    return pick, layers


def _module(k):
    return k.rsplit(".", 1)[0]


def _header(p):
    import struct
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h


def _check_skeleton(base, pick):
    """Every non-VQ tensor a band shard carries must match --base's dtype and
    shape (a different skeleton is a different model, not a mix)."""
    vq = _vq(base).union(*[_vq(s) for s in set(pick.values())])
    for f, src in pick.items():
        if src == base:
            continue
        hb, hs = _header(base / f), _header(src / f)
        for k, v in hs.items():
            if _module(k) in vq:
                continue
            w = hb.get(k)
            if w is None or (w["dtype"], w["shape"]) != (v["dtype"], v["shape"]):
                raise SystemExit(f"{src}/{f}: non-VQ tensor {k} differs from --base "
                                 f"({w and (w['dtype'], w['shape'])} vs {(v['dtype'], v['shape'])})")


def build(out, base, bands, copy=False, files_from=None):
    """files_from: where model.py, the non-map config keys and every other
    non-shard file come from (default --base). Use it when --base is an exact
    teacher that carries no VQ runtime: e.g. base=teacher, bands = the fitted
    halves, files_from = one fitted half."""
    ff = pathlib.Path(files_from) if files_from else base
    pick, _ = plan(base, bands)
    for src in {s for s in pick.values()} - {ff}:
        a, b = ff / "model.py", src / "model.py"
        if a.exists() and b.exists() and not filecmp.cmp(a, b, shallow=False):
            raise SystemExit(f"{src}/model.py differs from {ff.name}'s: one artifact, one runtime")
    _check_skeleton(base, pick)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.load(open(ff / "config.json"))
    maps = ("vq_modules", "quantization", "vq_skipzero")
    merged = {m: {} for m in maps}
    # keep non-module quantization keys (group_size, bits, mode) from base
    qtop = {k: v for k, v in json.load(open(base / "config.json")).get(
        "quantization", {}).items() if not isinstance(v, dict)}
    wm = {}
    for src in sorted({s for s in pick.values()}, key=str):
        si, sc = _idx(src), json.load(open(src / "config.json"))
        mine = {f for f, s in pick.items() if s == src}
        keys = {k: f for k, f in si.items() if f in mine}
        wm.update(keys)
        mods = {_module(k) for k in keys}
        for m in maps:
            for mod, e in (sc.get(m) or {}).items():
                if isinstance(e, dict) and mod in mods:
                    merged[m][mod] = e
        for f in mine:
            s, d = os.path.realpath(src / f), out / f
            if copy:
                r = subprocess.run(["cp", "-c", s, str(d)], capture_output=True)
                if r.returncode:
                    shutil.copy(s, d)
            else:
                os.symlink(s, d)
    # a module VQ in its source must not keep an affine entry from another
    for mod in merged["vq_modules"]:
        merged["quantization"].pop(mod, None)
    cfg["vq_modules"] = merged["vq_modules"]
    if "quantization" in cfg:
        cfg["quantization"] = {**qtop, **merged["quantization"]}
    if merged["vq_skipzero"] or "vq_skipzero" in cfg:
        cfg["vq_skipzero"] = merged["vq_skipzero"]
    total = sum(os.path.getsize(os.path.realpath(out / f)) for f in set(wm.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": wm}, indent=1))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ff.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in SKIP:
            shutil.copy(f, out)
    ks = collections.Counter(f"d{e.get('dim', e.get('d', '?'))}/K{e.get('k', '?')}"
                             for e in cfg["vq_modules"].values())
    print(f"{out.name}: {len(cfg['vq_modules'])} VQ modules {dict(ks)}, "
          f"{total / 2**30:.1f} GiB, {'copied' if copy else 'linked'}")
    return 0


def _band(s):
    src, rng = s.rsplit(":", 1)
    lo, hi = (int(x) for x in rng.split("-"))
    return pathlib.Path(src), lo, hi


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab mix", description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--band", action="append", default=[], type=_band,
                    help="SRC:LO-HI, repeatable; later bands win")
    ap.add_argument("--copy", action="store_true", help="real files instead of symlinks")
    ap.add_argument("--files-from", help="take model.py, config and other non-shard files from "
                                         "this source instead of --base (e.g. base = exact teacher)")
    ap.add_argument("--plan", action="store_true", help="print shard -> layers -> source and stop")
    a = ap.parse_args(argv)
    base = pathlib.Path(a.base)
    if a.plan:
        pick, layers = plan(base, a.band)
        for f in sorted(layers):
            print(f"{f:40s} layers {sorted(layers[f])}  <- {pick[f]}")
        return 0
    from vqlab import config as _cfg
    if a.copy:
        _cfg.require_free(a.out, sum(p.stat().st_size for p in base.glob("*.safetensors")), "mix --copy")
    return build(pathlib.Path(a.out), base, a.band, a.copy, a.files_from)


if __name__ == "__main__":
    sys.exit(main())
