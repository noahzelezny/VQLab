#!/usr/bin/env python3
"""mtp-arms: one pinned, servable arm per draft head, for a head A/B.

    vqlab mtp-arms --artifact <VQ trunk> --head q6=<file> --head vq=<file>
                   [--models-dir ~/.exo/models] [--prefix ab]

Each arm is a directory of SYMLINKS to the trunk's files (shards, config,
model.py, tokenizer) plus exactly ONE mtp-head*.safetensors, named
<prefix>-<tag>--<trunk name> under the Knurlogic models dir. The published
artifact is never touched: on 2026-10-02 the A/B swapped head files inside
the release folder itself, which is the live-artifact edit AGENTS.md forbids.
Then:

    vqlab speed-pair-knurlogic <arm a> <arm b> --machine ... --draft

alternates the arms, one fresh load each, and records acceptance beside the
speed. Distinct names keep Knurlogic from collapsing two arms that share
every shard into one identity.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab mtp-arms", description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--head", action="append", required=True, help="tag=path to a head file")
    ap.add_argument("--models-dir", default=os.environ.get("KNURLOGIC_MODELS", "~/.exo/models"))
    ap.add_argument("--prefix", default="ab")
    a = ap.parse_args(argv)
    art = pathlib.Path(a.artifact).resolve()
    md = pathlib.Path(a.models_dir).expanduser()
    names = []
    for spec in a.head:
        tag, _, path = spec.partition("=")
        head = pathlib.Path(path).resolve()
        if not head.is_file():
            raise SystemExit(f"no head file {head}")
        d = md / f"{a.prefix}-{tag}--{art.name}"
        if d.exists():
            raise SystemExit(f"{d} exists; pick another --prefix (arms are never overwritten)")
        d.mkdir(parents=True)
        for f in art.iterdir():
            if f.name.startswith("mtp-head") or f.name.startswith(".") or f.name == "__pycache__":
                continue
            os.symlink(f, d / f.name)
        os.symlink(head, d / "mtp-head.safetensors")
        names.append(d.name)
        print(f"{d.name}: trunk {art.name} + head {head.name}")
    print("\nnext: vqlab speed-pair-knurlogic " + " ".join(names[:2]) + " --machine <machine> --draft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
