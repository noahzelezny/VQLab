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
import hashlib
import os
import pathlib
import shlex
import subprocess
import sys

from vqlab.core.artifact import sizes

GIB = 2 ** 30
JUNK = (".DS_Store",)
LFS_MIN = 10 << 20          # files above this are LFS on the Hub


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


def _git_blob(p):
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hub_diff(art, repo, deep=False):
    """-> (changed_or_new, only_on_hub, repo_exists). Needs HF auth for
    private repos (HF_HOME / HF_TOKEN as for `hf`)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except RepositoryNotFoundError:
        return None, None, False
    hub = {s.rfilename: s for s in info.siblings}
    local = [p for p in sorted(art.rglob("*")) if p.is_file()
             and p.relative_to(art).as_posix() not in ()
             and not any(part == "__pycache__" or part.startswith(".") for part in p.relative_to(art).parts)
             and not p.name.endswith(".pyc")]
    changed = []
    for p in local:
        rel = p.relative_to(art).as_posix()
        h = hub.get(rel)
        if h is None:
            changed.append(rel)
            continue
        size = p.stat().st_size
        if h.size != size:
            changed.append(rel)
        elif size < LFS_MIN or h.lfs is None:
            if h.blob_id and _git_blob(p) != h.blob_id:
                changed.append(rel)
        elif deep and h.lfs.sha256 != _sha256(p):
            changed.append(rel)
    have = {p.relative_to(art).as_posix() for p in local}
    only_hub = sorted(k for k in hub if k not in have and k != ".gitattributes")
    return changed, only_hub, True


def _run(cmd):
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[2])}
    return subprocess.run([sys.executable, "-m", "vqlab.cli", *cmd], env=env).returncode


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
    files = "--all" if not exists else "--files " + " ".join(shlex.quote(f) for f in changed)
    print("\nREADY. Upload (a human runs this):\n")
    print(f"  vqlab publish --artifact {shlex.quote(str(art))} --repo {a.repo} {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
