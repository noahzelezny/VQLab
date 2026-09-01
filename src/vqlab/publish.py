#!/usr/bin/env python3
"""Upload an artifact to the Hub, and refuse to if the release gate fails.

    python -m vqlab.cli publish --artifact <dir> --repo <owner/name>
                                [--files model.py README.md | --all]
                                [--message TEXT] [--dry-run]

WHY THIS EXISTS. `check-release` was a gate anyone could forget, and on
2026-09-01 three dense 27B rungs went out with a bundled model.py importing a
module that exists only in our development venvs. Two of them could not
generate a single token for anyone who downloaded them. The gate would have
caught it; nothing made the gate run.

So the upload path itself now runs it. There is no --force and no --skip-gate:
a bypass that exists is a bypass that gets used at 2am, and the whole point is
that the promise not to forget is not something a person or an agent should
have to keep.

WHAT IT CHECKS BEYOND THE GATE. The gate certifies the bytes on disk at the
moment it runs. This hashes every file it is about to upload BEFORE and AFTER
the gate, and aborts if any of them moved in between -- because a bundled
model.py is generated from a working tree, and a working tree can have another
session editing it. That is not hypothetical: it nearly happened while this
was being written.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _digest(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab publish",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--repo", required=True, help="owner/name on the Hub")
    ap.add_argument("--files", nargs="+", default=None,
                    help="paths RELATIVE to the artifact dir. Default: the "
                         "runtime and card, which is what a fix usually "
                         "touches.")
    ap.add_argument("--all", action="store_true",
                    help="upload the whole directory (a first publish)")
    ap.add_argument("--message", default=None)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="run the gate and print the plan, upload nothing")
    a = ap.parse_args(argv)

    art = pathlib.Path(a.artifact).resolve()
    if not art.is_dir():
        print(f"FAIL: {art} is not a directory")
        return 1

    if a.all:
        targets = sorted(p for p in art.rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts)
    else:
        rels = a.files or ["model.py", "README.md"]
        targets = []
        import os
        for r in rels:
            # Containment is checked LEXICALLY, not via resolve(): resolve()
            # follows symlinks, so a legitimately symlinked artifact file
            # looks like an escape. The check exists to stop "../.." in the
            # argument, which normpath already settles.
            joined = os.path.normpath(os.path.join(str(art), r))
            if not (joined == str(art) or joined.startswith(str(art) + os.sep)):
                print(f"FAIL: {r} points outside the artifact directory")
                return 1
            p = pathlib.Path(joined)
            if not p.is_file():
                print(f"FAIL: {r} does not exist in {art}")
                return 1
            targets.append(p)

    print(f"artifact : {art}")
    print(f"repo     : {a.repo}")
    print(f"files    : {len(targets)}")
    for p in targets[:12]:
        print(f"   {p.relative_to(art)}  ({p.stat().st_size} bytes)")
    if len(targets) > 12:
        print(f"   ... and {len(targets) - 12} more")

    before = {p: _digest(p) for p in targets}

    print("\n--- release gate ---", flush=True)
    r = subprocess.run([sys.executable, str(HERE / "check_release.py"),
                        "--artifact", str(art),
                        "--max-tokens", str(a.max_tokens)])
    if r.returncode != 0:
        print("\nREFUSING TO UPLOAD: the release gate failed. Fix the "
              "artifact and run again. There is deliberately no override.")
        return 1

    moved = [p for p in targets if _digest(p) != before[p]]
    if moved:
        print("\nREFUSING TO UPLOAD: these files changed while the gate was "
              "running, so the gate did not certify what would be uploaded:")
        for p in moved:
            print(f"   {p.relative_to(art)}")
        return 1

    if a.dry_run:
        print("\n--dry-run: gate passed, nothing uploaded.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    msg = a.message or "Update artifact runtime/card (gated by vqlab publish)"
    print("\n--- uploading ---", flush=True)
    for p in targets:
        rel = str(p.relative_to(art))
        info = api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                               repo_id=a.repo, commit_message=msg)
        print(f"   {rel} -> {getattr(info, 'commit_url', info)}")
    print(f"\nPASS: {len(targets)} file(s) uploaded to {a.repo} after a "
          f"clean release gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
