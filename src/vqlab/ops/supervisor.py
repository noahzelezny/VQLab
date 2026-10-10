"""The detached process vqlab.ops.runs.launch starts: it takes the box's GPU
lease (a private lock for GPU-free commands), runs `vqlab <cmd>`, retries a
resumable command from its checkpoints, and records the outcome in meta.json.

    python -m vqlab.ops.supervisor --supervise RUN_DIR
"""

from __future__ import annotations

import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vqlab.core.gpu_lease import acquire_lease, lease_path, release_lease
from vqlab.ops.roots import REPO
from vqlab.ops.runs import GPU_FREE, _read_meta


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write(run_dir: Path, meta: Dict[str, Any]) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def _attempts(run_dir: Path, meta: Dict[str, Any], argv: List[str],
              st: Dict[str, int]) -> None:
    """Run argv until it exits 0 or the retries are spent; st carries rc and
    attempts (updated in place, so a SIGTERM mid-sleep still records them)."""
    with open(run_dir / "job.log", "ab") as log:
        while st["attempts"] <= int(meta.get("retries", 0)):
            st["attempts"] += 1
            attempts = st["attempts"]
            log.write(f"\n=== attempt {attempts} {_now()}: {' '.join(shlex.quote(a) for a in argv)}\n".encode())
            log.flush()
            meta.update(status="running", attempt=attempts, started=meta.get("started") or _now())
            _write(run_dir, meta)
            try:
                proc = subprocess.Popen(argv, cwd=meta.get("repo", str(REPO)), stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
                meta["job_pid"] = proc.pid
                _write(run_dir, meta)
                st["rc"] = rc = proc.wait()
            except KeyboardInterrupt:
                try:
                    proc.terminate()
                    proc.wait(30)
                except Exception:
                    pass
                st["rc"] = -15
                return
            if rc == 0:
                break
            log.write(f"=== exit {rc}\n".encode())
            log.flush()
            if attempts <= int(meta.get("retries", 0)):
                time.sleep(60)


def supervise(run_dir: Path) -> int:
    meta = _read_meta(run_dir)
    # A GPU-free job locks a private file: same supervisor code path, no
    # contention with the shared GPU lease.
    lp = run_dir / "no-gpu.lock" if meta.get("cmd") in GPU_FREE else lease_path()
    fd = acquire_lease(lp, f"vqlab:{meta.get('cmd')}", meta.get("run_id"), wait_s=1800)
    if fd is None:
        meta.update(status="deferred", finished=_now(),
                    note="GPU lease held for 30 min; deferred, not failed")
        _write(run_dir, meta)
        return 0

    argv = [sys.executable, "-m", "vqlab.cli", meta["cmd"], *meta.get("args", [])]
    st = {"rc": 1, "attempts": 0}
    stopping = {"flag": False}

    def _term(_s, _f):
        stopping["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    try:
        _attempts(run_dir, meta, argv, st)
    finally:
        rc, attempts = st["rc"], st["attempts"]
        status = "stopped" if stopping["flag"] else ("completed" if rc == 0 else "failed")
        meta.update(status=status, exit_code=rc, attempts=attempts, finished=_now())
        _write(run_dir, meta)
        release_lease(fd)
    return 0 if st["rc"] == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m vqlab.ops.supervisor")
    ap.add_argument("--supervise", metavar="RUN_DIR", required=True)
    return supervise(Path(ap.parse_args(argv).supervise))


if __name__ == "__main__":
    sys.exit(main())
