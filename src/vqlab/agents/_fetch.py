"""One GUI download's own process (agents/downloads.py starts it):

    python -m vqlab.agents._fetch REPO DEST LOCKFILE RCFILE PATTERN...

`huggingface_hub.snapshot_download` of REPO into DEST, only the files the
PATTERNs allow. The process holds a flock on DEST/LOCKFILE for its whole
life, with its pid in it: the server signals a recorded pid only while that
lock says it is still this fetch (a pid alone can be reused after a
restart). A second fetch of the same folder exits 2 and touches nothing. It
records how it ended in DEST/RCFILE: 0 done, 1 failed (its stderr log says
why). A stopped or interrupted fetch (SIGTERM, ctrl-c) records nothing.
"""
from __future__ import annotations

import fcntl
import os
import pathlib
import sys

from huggingface_hub import snapshot_download


def main(argv: list[str]) -> int:
    repo, dest, lockfile, rcfile, *allow = argv
    d = pathlib.Path(dest)
    lk = open(d / lockfile, "a+")         # held open until exit: the lock is the liveness
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # another fetch owns this folder: its pid and exit file stay its own
        print(f"another fetch of {d} is running; not starting a second", file=sys.stderr)
        return 2
    lk.truncate(0)
    lk.write(str(os.getpid()))
    lk.flush()
    try:
        snapshot_download(repo, local_dir=d, allow_patterns=allow)
    except Exception:
        (d / rcfile).write_text("1")
        raise
    (d / rcfile).write_text("0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
