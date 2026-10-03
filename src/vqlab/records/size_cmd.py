#!/usr/bin/env python3
"""vqlab size: an artifact's size three ways, from its own headers.

    vqlab size <artifact> [--json]

text (the headline every card quotes), + vision tower, + MTP head (indexed
or sidecar), and the full download. The 397B indexes its MTP head while
GLM and DeepSeek ship it as a sidecar, so "index total" is not "text".
"""
from __future__ import annotations

import argparse
import json
import sys

from vqlab.core.artifact import sizes

GIB = 2 ** 30


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab size", description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check-index", action="store_true",
                    help="exit 1 if the index's metadata.total_size is missing or differs from the "
                         "shards' tensor bytes (loaders size and place models by it)")
    ap.add_argument("--fix-index", action="store_true",
                    help="rewrite metadata.total_size from the shards (the weight_map is unchanged); "
                         "a shipped artifact then needs its index re-published")
    a = ap.parse_args(argv)
    if a.check_index or a.fix_index:
        import pathlib
        from vqlab.core.artifact import Artifact, tensor_bytes, write_index
        art = Artifact.open(a.artifact)
        p = pathlib.Path(a.artifact) / "model.safetensors.index.json"
        doc = json.loads(p.read_text())
        have = (doc.get("metadata") or {}).get("total_size")
        want = tensor_bytes(art.dir, art.shards)
        state = "ok" if have == want else ("missing" if have is None else
                                           f"stale by {(have - want) / GIB:+.2f} GiB")
        print(f"index total_size {have} vs tensor bytes {want}: {state}")
        if have != want and a.fix_index:
            md = {k: v for k, v in (doc.get("metadata") or {}).items() if k != "total_size"}
            write_index(art.dir, art.index, total_size=want, metadata=md)
            print(f"rewrote {p.name} (total_size {want})")
            from vqlab.records import provenance
            if (art.dir / provenance.RECORD).exists():
                # an in-place edit is an AMENDMENT: the prior record goes to
                # history, and this one says exactly what changed
                provenance.write_build_record(
                    art.dir, tool="size --fix-index",
                    argv=["size", str(art.dir), "--fix-index"],
                    method={"recorded_by": "size --fix-index",
                            "note": f"index metadata.total_size {have} -> {want} (tensor bytes); "
                                    "weight_map and every shard unchanged"},
                    full_hash={"model.safetensors.index.json"})
                print("build record amended (prior record kept in history)")
            return 0
        if a.fix_index:
            print(f"--fix-index: {p.name} already correct; nothing rewritten")
            return 1                      # a writer that changed nothing
        return 0 if have == want else 1
    s = sizes(a.artifact)
    if a.json:
        print(json.dumps(s, indent=1))
        return 0
    mtp = s["mtp"] + s["mtp_sidecar"]
    print(f"text weights      {s['text'] / GIB:8.2f} GiB")
    print(f"+ vision tower    {(s['text'] + s['tower']) / GIB:8.2f} GiB   (tower {s['tower'] / GIB:.2f})")
    print(f"+ MTP head        {(s['text'] + s['tower'] + mtp) / GIB:8.2f} GiB   (head {mtp / GIB:.2f}"
          + (f", sidecar {', '.join(s['mtp_files'])}" if s["mtp_files"] else ", indexed" if s["mtp"] else "")
          + ")")
    print(f"full download     {s['download'] / GIB:8.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
