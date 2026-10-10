#!/usr/bin/env python3
"""release-prep: everything before an upload, in one command, ending in the
exact upload line for a human to run (structural feedback #6).

    vqlab release-prep <artifact> --repo owner/name [--no-smoke] [--hash]

1. sizes three ways (`vqlab size`): the numbers the card must quote
2. junk files that must not ship (.DS_Store, ._*, __pycache__, *.pyc)
3. the build record (`vqlab provenance --verify`)
4. the release gate (`vqlab check-release`, smoke included unless --no-smoke)
5. a diff against the Hub: which files are new or changed. Small files are
   compared by git blob hash (exact, nothing downloaded); weight files by
   size, or by sha256 against the Hub's LFS hash with --hash (reads every
   byte locally; minutes per 100 GiB).
6. prints `vqlab publish ... --files <changed>` (publish re-runs the gate and
   hashes before/after). Uploading the WHOLE folder hashed 1.5 TB to send a
   few kilobytes on 2026-10-01; this names only what changed.

Nothing here uploads: publishing is a human's action.
"""
from __future__ import annotations

import argparse
import importlib.resources
import os
import pathlib
import shlex
import subprocess
import sys

from vqlab.core.artifact import sizes
from vqlab.core.hub import LFS_MIN, hub_diff  # noqa: F401  (re-exported)
from vqlab.util.hashing import git_blob_sha1, sha256_file

_git_blob = git_blob_sha1   # old private names, kept for callers/tests
_sha256 = sha256_file

GIB = 2 ** 30
JUNK = (".DS_Store",)


def _junk(d):
    out = []
    for root, dirs, files in os.walk(d):
        for x in dirs:
            if x == "__pycache__":
                out.append(os.path.join(root, x))
        for f in files:
            if f in JUNK or f.startswith("._") or f.endswith(".pyc"):
                out.append(os.path.join(root, f))
    return out


def _run(cmd):
    # the directory holding THIS vqlab package, so the child CLI runs the
    # same tree (a checkout's src/, or site-packages when installed)
    pkg = pathlib.Path(str(importlib.resources.files("vqlab"))).resolve()
    env = {**os.environ, "PYTHONPATH": str(pkg.parent)}
    return subprocess.run([sys.executable, "-m", "vqlab.cli", *cmd], env=env).returncode


def publish_line(art, repo, changed, exists) -> str:
    """The exact upload line a human runs: only what changed, or --all for a
    first release. Nothing here uploads."""
    files = "--all" if not exists else "--files " + " ".join(shlex.quote(f) for f in changed)
    return f"vqlab publish --artifact {shlex.quote(str(art))} --repo {repo} {files}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab release-prep", description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--repo", required=True, help="owner/name on the Hub")
    ap.add_argument("--no-smoke", action="store_true", help="passed to check-release")
    ap.add_argument("--hash", action="store_true", help="compare weight files by sha256, not size")
    ap.add_argument("--skip-gate", action="store_true",
                    help="diff and sizes only (publish runs the gate anyway)")
    a = ap.parse_args(argv)
    art = pathlib.Path(a.artifact)
    problems = []

    s = sizes(art)
    mtp = s["mtp"] + s["mtp_sidecar"]
    print(f"== sizes: text {s['text'] / GIB:.2f} GiB | +tower {(s['text'] + s['tower']) / GIB:.2f} | "
          f"+MTP {(s['text'] + s['tower'] + mtp) / GIB:.2f} | download {s['download'] / GIB:.2f}")

    junk = _junk(art)
    print(f"== junk: {len(junk)} local-only file(s), never uploaded (publish names files explicitly)"
          + ("" if not junk else ": " + ", ".join(os.path.relpath(j, art) for j in junk[:5])))

    if not a.skip_gate:
        print("== provenance --verify", flush=True)
        if _run(["provenance", "--verify", str(art)]):
            problems.append("build record does not verify")
        print("== check-release", flush=True)
        if _run(["check-release", "--artifact", str(art)] + (["--no-smoke"] if a.no_smoke else [])):
            problems.append("check-release failed")

    print(f"== Hub diff against {a.repo}", flush=True)
    changed, only_hub, exists = hub_diff(art, a.repo, a.hash)
    if not exists:
        print("   repo does not exist yet: first release uploads everything")
    else:
        print(f"   {len(changed)} new/changed: {', '.join(changed) or '-'}")
        if only_hub:
            print(f"   on the Hub but not here (left alone by publish): {', '.join(only_hub)}")
        if not a.hash:
            print("   weight files compared by SIZE; --hash compares sha256")

    if problems:
        print("\nNOT READY:\n  - " + "\n  - ".join(problems))
        return 1
    if exists and not changed:
        print("\nREADY, and the Hub already matches: nothing to upload.")
        return 0
    print("\nREADY. Upload (a human runs this):\n")
    print(f"  {publish_line(art, a.repo, changed, exists)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
