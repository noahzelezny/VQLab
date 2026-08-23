#!/usr/bin/env python3
"""Provenance manifest for artifacts. WRITE-ONLY on the manifest file itself —
this tool never modifies an artifact.

WHY. On 2026-08-21 two artifacts were silently overwritten in place and both
were caught by chance mtime checks, not by anything failing:
  - E94's SCORED 35B artifact, by a refit aimed at its own --out path. Its
    published number described bytes that no longer existed.
  - the 397B base struct6-tail3x3, rewritten Aug 19 — three days AFTER the
    artifact built from it — which silently invalidated every 397B refit
    comparison since, and was only found by looking at `ls -l` after four
    in-algorithm explanations had been falsified.
Artifacts carry no provenance stamp, so a rerun aimed at an existing path is
indistinguishable from the original. This converts "was this overwritten?"
from forensics into a lookup.

    ./artifact_manifest.py write <dir> [<dir> ...]   # stamp
    ./artifact_manifest.py check <dir> [<dir> ...]   # verify against the stamp

Verified per III.5 before use: FAILS (exit 2) on a rewritten shard, PASSES on
an untouched one. Known limitation: mtime is recorded at second granularity, so
a touch and a rewrite within the same second are distinguished by the hash, not
the timestamp — which is the right way round.

Records per shard: name, bytes, mtime, and sha256 of the FIRST 1 MiB (full
hashing 400 GiB is not worth the wall time; the header plus first tensors move
on any real rewrite). Manifest lives OUTSIDE the artifact, in manifests/, so
stamping cannot alter the bytes it describes.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import os
MAN = pathlib.Path(os.environ.get("MOEMASH_MANIFEST_DIR", "manifests"))


def head_sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def snapshot(d):
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        st = f.stat()
        out[f.name] = {"bytes": st.st_size, "mtime": int(st.st_mtime),
                       "head_sha256": head_sha(f)}
    return out


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("write", "check"))
ap.add_argument("dirs", nargs="+")
a = ap.parse_args()
MAN.mkdir(exist_ok=True)
rc = 0

for dd in a.dirs:
    d = pathlib.Path(dd)
    if not d.is_dir():
        print(f"SKIP {dd}: not a directory"); continue
    mf = MAN / (d.name + ".json")
    cur = snapshot(d)
    if a.mode == "write":
        mf.write_text(json.dumps({"artifact": str(d), "shards": cur}, indent=1))
        print(f"STAMPED {d.name}: {len(cur)} shards -> {mf.name}")
        continue
    if not mf.exists():
        print(f"NO MANIFEST {d.name} — cannot verify; stamp it now if these "
              f"bytes are the ones you mean to keep")
        rc = max(rc, 1); continue
    old = json.load(open(mf))["shards"]
    added = sorted(set(cur) - set(old))
    gone = sorted(set(old) - set(cur))
    moved = [k for k in sorted(set(cur) & set(old))
             if cur[k]["bytes"] != old[k]["bytes"]
             or cur[k]["head_sha256"] != old[k]["head_sha256"]]
    touched = [k for k in sorted(set(cur) & set(old))
               if k not in moved and cur[k]["mtime"] != old[k]["mtime"]]
    if added or gone or moved:
        print(f"CHANGED {d.name}: {len(moved)} rewritten, {len(added)} added, "
              f"{len(gone)} removed")
        for k in (moved + added + gone)[:5]:
            print(f"    {k}")
        rc = 2
    elif touched:
        print(f"ok {d.name}: bytes identical, {len(touched)} shard mtimes moved "
              f"(a copy or touch, not a rewrite)")
    else:
        print(f"ok {d.name}: {len(cur)} shards unchanged")
sys.exit(rc)
