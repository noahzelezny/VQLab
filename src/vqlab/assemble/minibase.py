#!/usr/bin/env python3
"""minibase: a fit base holding only the shards of one layer band (symlinks).

    vqlab minibase --base <dir> --layers LO-HI --out <dir>

fit-moe rewrites EVERY base shard, even with a --vq-layers subset (a one-band
fit of a 150 GiB base writes 150 GiB). A minibase restricts the base to the
shards holding that band's layers, so a band fit writes only those. The
index lists only those shards' tensors; config and other files are copied.
Output is a fit INPUT, not an artifact: assemble the bands with `vqlab mix`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab minibase", description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--layers", required=True, help="LO-HI")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base, out = pathlib.Path(a.base), pathlib.Path(a.out)
    lo, hi = (int(x) for x in a.layers.split("-"))
    wm = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    keep = set()
    for k, f in wm.items():
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", k)
        if m and lo <= int(m.group(1)) <= hi:
            keep.add(f)
    if not keep:
        raise SystemExit(f"no shard holds layers {lo}-{hi}")
    out.mkdir(parents=True, exist_ok=False)
    sub = {k: f for k, f in wm.items() if f in keep}
    for f in keep:
        os.symlink(os.path.realpath(base / f), out / f)
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": sub}, indent=1))
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in (
                "model.safetensors.index.json", "vqlab_provenance.json"):
            shutil.copy(f, out)
    other = sorted({int(m.group(1)) for k in sub for m in [re.search(r"(?:^|\.)layers\.(\d+)\.", k)]
                    if m and not lo <= int(m.group(1)) <= hi})
    print(f"{out.name}: {len(keep)} shards, {len(sub)} tensors for layers {lo}-{hi}"
          + (f" (shards also hold layers {other})" if other else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
