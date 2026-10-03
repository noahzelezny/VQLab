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
import pathlib
import sys

from vqlab.core.artifact import (Artifact, copy_other_files, merge_maps, module_of,
                                 place, write_config, write_index)


def plan(base, bands):
    """shard -> source Artifact. A band takes a shard when every VQ module
    layer in that shard lies inside the band; only VQ modules count (the rest
    of a shard is skeleton, identical across sources, checked by header)."""
    base = Artifact.open(base)
    srcs = [(Artifact.open(s), lo, hi) for s, lo, hi in bands]
    vq = base.vq_modules.union(*[a.vq_modules for a, _, _ in srcs])
    layers = base.shard_layers(vq)
    pick = {f: base for f in layers}
    for a, lo, hi in srcs:
        want = set(range(lo, hi + 1))
        slayers = a.shard_layers(vq)
        for f, ls in layers.items():
            if ls & want:
                if not ls <= want:
                    raise SystemExit(f"{f} holds layers {sorted(ls)}: straddles band "
                                     f"{lo}-{hi}; align the band to shard boundaries")
                if slayers.get(f) != ls:
                    raise SystemExit(f"{a.dir}: shard {f} is missing or holds other layers "
                                     f"({sorted(slayers.get(f, []))} vs {sorted(ls)}); "
                                     "mix needs the same skeleton")
                pick[f] = a
    return pick, layers


def _check_skeleton(base, pick, vq):
    """Every non-VQ tensor a band shard carries must match --base's dtype and
    shape (a different skeleton is a different model, not a mix)."""
    for f, src in pick.items():
        if src is base:
            continue
        hb, hs = base.header(f), src.header(f)
        for k, v in hs.items():
            if module_of(k) in vq:
                continue
            w = hb.get(k)
            if w is None or (w["dtype"], w["shape"]) != (v["dtype"], v["shape"]):
                raise SystemExit(f"{src.dir}/{f}: non-VQ tensor {k} differs from --base "
                                 f"({w and (w['dtype'], w['shape'])} vs {(v['dtype'], v['shape'])})")


def build(out, base, bands, copy=False, files_from=None):
    """files_from: where model.py, the non-map config keys and every other
    non-shard file come from (default --base). Use it when --base is an exact
    teacher that carries no VQ runtime: e.g. base=teacher, bands = the fitted
    halves, files_from = one fitted half."""
    out = pathlib.Path(out)
    pick, _ = plan(base, bands)
    base = Artifact.open(base)
    ff = Artifact.open(files_from) if files_from else base
    for src in {a.dir: a for a in pick.values()}.values():
        if src.dir == ff.dir:
            continue
        a, b = ff.dir / "model.py", src.dir / "model.py"
        if a.exists() and b.exists() and not filecmp.cmp(a, b, shallow=False):
            raise SystemExit(f"{src.dir}/model.py differs from {ff.dir.name}'s: one artifact, one runtime")
    vq = base.vq_modules.union(*[a.vq_modules for a in pick.values()])
    _check_skeleton(base, pick, vq)
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(ff.config)
    qtop = {k: v for k, v in base.config.get("quantization", {}).items() if not isinstance(v, dict)}
    wm, sources = {}, []
    for src in sorted({a.dir: a for a in pick.values()}.values(), key=lambda a: str(a.dir)):
        mine = {f for f, a in pick.items() if a.dir == src.dir}
        keys = {k: f for k, f in src.index.items() if f in mine}
        wm.update(keys)
        sources.append((src, {module_of(k) for k in keys}))
        for f in mine:
            place(src.shard_path(f), out / f, copy)
    merged = merge_maps(sources)
    cfg["vq_modules"] = merged["vq_modules"]
    if "quantization" in cfg:
        cfg["quantization"] = {**qtop, **merged["quantization"]}
    if merged["vq_skipzero"] or "vq_skipzero" in cfg:
        cfg["vq_skipzero"] = merged["vq_skipzero"]
    write_index(out, wm)
    total = sum((out / f).stat().st_size for f in set(wm.values()))   # follows symlinks
    write_config(out, cfg)
    copy_other_files(ff, out)
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
            print(f"{f:40s} layers {sorted(layers[f])}  <- {pick[f].dir}")
        return 0
    from vqlab import config as _cfg
    if a.copy:
        _cfg.require_free(a.out, sum(p.stat().st_size for p in base.glob("*.safetensors")), "mix --copy")
    return build(pathlib.Path(a.out), base, a.band, a.copy, a.files_from)


if __name__ == "__main__":
    sys.exit(main())
