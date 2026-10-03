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
import os
import pathlib
import sys

from vqlab.core.artifact import Artifact, copy_other_files, layer_of, write_index


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab minibase", description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--layers", required=True, help="LO-HI")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base, out = Artifact.open(a.base), pathlib.Path(a.out)
    lo, hi = (int(x) for x in a.layers.split("-"))
    keep = {f for k, f in base.index.items() if lo <= layer_of(k) <= hi}
    if not keep:
        raise SystemExit(f"no shard holds layers {lo}-{hi}")
    out.mkdir(parents=True, exist_ok=False)
    sub = {k: f for k, f in base.index.items() if f in keep}
    for f in keep:
        os.symlink(base.shard_path(f), out / f)
    write_index(out, sub)
    (out / "config.json").write_text((base.dir / "config.json").read_text())
    copy_other_files(base, out)
    other = sorted({layer_of(k) for k in sub} - set(range(lo, hi + 1)) - {-1})
    print(f"{out.name}: {len(keep)} shards, {len(sub)} tensors for layers {lo}-{hi}"
          + (f" (shards also hold layers {other})" if other else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
