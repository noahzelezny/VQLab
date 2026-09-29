#!/usr/bin/env python3
"""Paper v5 regression: does today's scorer + runtime still produce the paper?

    python research/paper/regress.py            # GPU, ~5 min, nothing else placed

Re-scores the 27B VQ-3.9bpw pin on the prose cache through `vqlab kl-ladder`
into a fresh directory and fails unless mean KL is exactly the published
146.5634 mnats (scoring is deterministic to every printed digit). Run it
before any release that touches the scorer, the VQ runtime or mlx. The
cheap half -- that RESULTS-V5.md still regenerates from the saved arrays --
is in `vqlab selftest`.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

PR = pathlib.Path("<scratch>")
ARTIFACT = PR / "paper_rev" / "pinv2" / "q27_3.9"
CACHE = PR / "teacher_caches_full" / "q27_prose_12k"
EXPECT = 146.5634
SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def main() -> int:
    for p in (ARTIFACT, CACHE):
        if not p.exists():
            print(f"FAIL: {p} is not here")
            return 1
    with tempfile.TemporaryDirectory(dir=PR) as tmp:
        out = pathlib.Path(tmp) / "ladder.json"
        r = subprocess.run([sys.executable, "-m", "vqlab.cli", "kl-ladder",
                            "--cache", f"prose={CACHE}", "--rung", f"r39={ARTIFACT}",
                            "--per-pos-dir", tmp, "--out", str(out)],
                           env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
        if r.returncode != 0 or not out.exists():
            print(f"FAIL: kl-ladder rc={r.returncode}")
            return 1
        got = round(json.loads(out.read_text())["table"]["r39"]["prose"]["mean_kl_millinats"], 4)
    ok = got == EXPECT
    print(f"{'PASS' if ok else 'FAIL'}: 27B VQ-3.9bpw prose KL {got} (paper {EXPECT})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
