#!/usr/bin/env python3
"""Compose a MIXED per-layer-per-projection promotion in one splice pass.

WHY A SEPARATE TOOL. `v2_sweep.py --projections` promotes the SAME subset
on every layer in a --combo, and its preflight refuses a non-flat base, so
it cannot chain eleven different subsets. The knapsack solution needs
exactly that: L43 up+down, L45 all three, L37 gate only, and so on.

The splice contract is identical to v2_sweep's (whole-tensor copy of the
donor's codes/codebook/vq_scales, per-module vq_modules entry, index
rebuild); this just accepts a per-layer spec and does it in one pass, so
untouched shards are hardlinked exactly once.

  ./mix_build.py --base <2.2 dir> --donor <2.4 dir> --out <dir> \
      --spec "43:up,down 45:all 37:gate"
"""
import argparse
import json
import os
import pathlib
import shutil

import mlx.core as mx

GiB = 2 ** 30
ALL = ("gate_proj", "up_proj", "down_proj")
SUF = ("codes", "codebook", "vq_scales")


def parse_spec(spec):
    out = {}
    for part in spec.split():
        layer, _, projs = part.partition(":")
        L = int(layer)
        if projs in ("all", "ALL3", ""):
            sel = ALL
        else:
            sel = tuple(p if p.endswith("_proj") else p + "_proj"
                        for p in projs.split(","))
        bad = [p for p in sel if p not in ALL]
        if bad:
            raise SystemExit(f"unknown projection(s) {bad} in {part!r}")
        out[L] = sel
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True,
                    help='e.g. "43:up,down 45:all 37:gate"')
    a = ap.parse_args()

    spec = parse_spec(a.spec)
    bp, dp, out = (pathlib.Path(x) for x in (a.base, a.donor, a.out))
    idx = json.load(open(bp / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    cfg = json.load(open(bp / "config.json"))
    dcfg = json.load(open(dp / "config.json"))
    dwm = json.load(open(dp / "model.safetensors.index.json"))["weight_map"]

    promoted, touched = set(), set()
    for L, projs in sorted(spec.items()):
        for p in projs:
            m = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}"
            if m not in cfg["vq_modules"] or m not in dcfg["vq_modules"]:
                raise SystemExit(f"{m} missing from base or donor vq_modules")
            if wm[m + ".codes"] != dwm[m + ".codes"]:
                raise SystemExit(f"shard mismatch for {m}; per-key rewrite needed")
            promoted.add(m)
            touched.update(wm[m + "." + s] for s in SUF)
    units = sum(len(v) for v in spec.values())
    print(f"promoting {len(promoted)} modules across {len(spec)} layers "
          f"({units} projection-units), rewriting {len(touched)} shard(s)",
          flush=True)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # non-shard files come across as-is. A .safetensors file that the INDEX
    # does not reference is a SIDECAR (mtp-head-q6, vision grafts), not a
    # shard -- skipping every .safetensors dropped the 397B's published
    # 5.4 GiB MTP head from a rebuild (caught 2026-09-07, before it could
    # be uploaded over the live repo).
    _idx_shards = set(wm.values())
    for f in bp.iterdir():
        if f.is_dir():
            continue
        if f.suffix == ".safetensors" and f.name in _idx_shards:
            continue
        shutil.copy2(f, out / f.name)

    mx.set_default_device(mx.cpu)
    for sh in sorted(set(wm.values())):
        if sh not in touched:
            os.link(bp / sh, out / sh)
            continue
        bt = mx.load(str(bp / sh))
        dt = mx.load(str(dp / sh))
        n = 0
        for name in list(bt):
            mod, _, suf = name.rpartition(".")
            if mod in promoted and suf in SUF:
                bt[name] = dt[name]
                n += 1
        tmp = out / (sh[:-len(".safetensors")] + ".tmp.safetensors")
        mx.save_safetensors(str(tmp), bt)
        os.replace(tmp, out / sh)
        del bt, dt
        print(f"    rewrote {sh}: {n} tensors", flush=True)

    for m in promoted:
        cfg["vq_modules"][m] = dcfg["vq_modules"][m]
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    idx.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(out / s) for s in sorted(set(wm.values())))
    json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=1)
    got = idx["metadata"]["total_size"] / GiB
    base_gib = json.load(open(bp / "model.safetensors.index.json"))["metadata"]["total_size"] / GiB
    print(f"DONE {out.name}: {got:.3f} GiB ({got - base_gib:+.3f} vs base)",
          flush=True)


if __name__ == "__main__":
    main()
