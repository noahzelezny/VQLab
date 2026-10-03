#!/usr/bin/env python3
"""vqlab scratch: what on the scratch volume a human could delete, and why.

    vqlab scratch reclaimable [--root DIR ...] [--full] [--json]

Lists unpacked fits and other intermediate artifact dirs under the configured
scratch roots (`vqlab.config`) whose PACKED counterpart exists and verifies,
with sizes and a total, and PRINTS the `rm -rf` lines. It never deletes
anything: deletion is the human's action, and there is no --delete flag.

The SSD went from 482 GiB free to 15 GiB in one night because unpacked fits
sat beside their packed copies (lab/docs/OPERATOR-NOTES-2026-10-03.md, sec 3).
Being sure a dir is redundant is what the build records are for: a packed
artifact's record (`vqlab pack` / `pack-dense`, written by the CLI) names its
`src` input with that input's provenance id and per-shard fingerprint. A
source is listed only when ALL of these hold:

  * the packed artifact's record is intact (its id still hashes) and its
    outputs still match it (sizes; `--full` re-hashes every byte);
  * the source still IS what was packed: its own record id equals the one
    the packed record names, and its shard fingerprint (bytes + first-MiB
    sha256 per shard) is unchanged;
  * nothing else under the roots resolves a symlink INTO the source (a pin,
    a minibase or a mix that would break when it goes).

Everything else is reported as KEPT with the reason, so an absence is never
silent.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import sys

from vqlab import config as C
from vqlab.records import provenance

GIB = 2 ** 30
# tool name in a build record -> the input role that names what it packed
PACKERS = {"pack": "src", "pack-dense": "src"}
MAX_DEPTH = 4


def _artifacts(roots):
    """Every dir under `roots` carrying a build record (no descent into an
    artifact's own subdirs past MAX_DEPTH)."""
    seen = set()
    for root in roots:
        root = pathlib.Path(root)
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            d = pathlib.Path(dirpath)
            depth = len(d.relative_to(root).parts)
            dirnames[:] = [n for n in dirnames if not n.startswith(".")] if depth < MAX_DEPTH else []
            if provenance.RECORD in filenames:
                rp = os.path.realpath(d)
                if rp not in seen:
                    seen.add(rp)
                    yield d


def _dir_bytes(d: pathlib.Path) -> int:
    """Bytes deleting `d` frees: regular files only (a symlinked shard is
    another artifact's bytes)."""
    n = 0
    for dirpath, _, filenames in os.walk(d):
        for f in filenames:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                n += os.lstat(p).st_size
    return n


def _links_into(roots, target: pathlib.Path):
    """Symlinks under `roots` (outside `target` itself) that resolve into it."""
    t = os.path.realpath(target)
    hits = []
    for root in roots:
        if not pathlib.Path(root).is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.realpath(dirpath) == t:
                dirnames[:] = []
                continue
            for f in filenames + dirnames:
                p = os.path.join(dirpath, f)
                if os.path.islink(p):
                    rp = os.path.realpath(p)
                    if rp == t or rp.startswith(t + os.sep):
                        hits.append(p)
    return hits


def _packed_ok(art: pathlib.Path, rec: dict, full: bool):
    """None if the packed artifact still matches its record, else why not."""
    if full:
        bad = provenance.verify(art, rec)
        return f"{bad[0][0]}: {bad[0][1]}" if bad else None
    import hashlib
    if hashlib.sha256(provenance._canon(rec).encode()).hexdigest() != rec.get("id"):
        return "record edited after it was written (id mismatch)"
    for name, want in rec.get("outputs", {}).items():
        f = art / name
        if name in provenance.JUNK or name.startswith("._"):
            continue
        if not f.exists():
            return f"{name}: missing"
        if os.path.getsize(os.path.realpath(f)) != want["bytes"]:
            return f"{name}: bytes differ from its record"
    return None


def reclaimable(roots, full=False):
    """-> (candidates, kept): candidates [{path, bytes, packed, tool, record}],
    kept [{path, packed, reason}]."""
    roots = [pathlib.Path(r) for r in roots]
    rroots = [os.path.realpath(r) for r in roots]
    under = lambda p: any(p == r or p.startswith(r + os.sep) for r in rroots)  # noqa: E731
    cands, kept, claimed = [], [], set()
    for art in _artifacts(roots):
        try:
            rec = provenance.load(art)
        except (OSError, ValueError):
            continue
        tool = (rec.get("tool") or {}).get("name")
        role = PACKERS.get(tool)
        if not role:
            continue
        for inp in rec.get("inputs", []):
            if inp.get("role") != role:
                continue
            src = pathlib.Path(inp.get("realpath") or inp.get("path", ""))
            rsrc = os.path.realpath(src)
            if not under(rsrc) or rsrc in claimed or rsrc == os.path.realpath(art):
                continue
            row = {"path": str(src), "packed": str(art), "tool": tool, "record": rec["id"]}
            if not src.is_dir():
                continue                                 # already gone
            why = _packed_ok(art, rec, full)
            if why:
                kept.append({**row, "reason": f"packed artifact does not verify ({why})"})
                continue
            have = provenance.fingerprint(src)
            if inp.get("provenance_id") and have.get("provenance_id") != inp["provenance_id"]:
                kept.append({**row, "reason": "source's build record changed since it was packed"})
                continue
            if have.get("shards") != inp.get("shards"):
                kept.append({**row, "reason": "source shards changed since they were packed"})
                continue
            links = _links_into(roots, src)
            if links:
                kept.append({**row, "reason": f"{len(links)} symlink(s) resolve into it, e.g. {links[0]}"})
                continue
            claimed.add(rsrc)
            cands.append({**row, "bytes": _dir_bytes(src)})
    return cands, kept


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab scratch", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reclaimable", help="list unpacked dirs whose verified pack exists; print the rm lines")
    r.add_argument("--root", action="append", default=None,
                   help="scan this dir instead of the configured scratch (repeatable)")
    r.add_argument("--full", action="store_true",
                   help="re-hash every output of each packed artifact against its record "
                        "(default: record integrity + sizes, which reads no shard bytes)")
    r.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    roots = a.root or [C.scratch()]
    cands, kept = reclaimable(roots, full=a.full)
    total = sum(c["bytes"] for c in cands)
    if a.json:
        print(json.dumps({"roots": [str(p) for p in roots], "reclaimable": cands,
                          "kept": kept, "total_bytes": total}, indent=1))
        return 0
    print(f"scratch roots: {', '.join(map(str, roots))}")
    if not cands:
        print("nothing reclaimable: no unpacked dir has a verified packed counterpart")
    for c in sorted(cands, key=lambda c: -c["bytes"]):
        print(f"  {c['bytes'] / GIB:9.2f} GiB  {c['path']}\n"
              f"               packed -> {c['packed']}  ({c['tool']}, record {c['record'][:12]})")
    for k in kept:
        print(f"  KEPT  {k['path']}\n        {k['reason']}  (packed {k['packed']})")
    if cands:
        print(f"total {total / GIB:.2f} GiB in {len(cands)} dir(s)\n")
        print("# Nothing was deleted. To reclaim the space, a human runs:")
        for c in cands:
            print(f"rm -rf {shlex.quote(c['path'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
