#!/usr/bin/env python3
"""HARD GUARD: refuse any resident-memory operation on a model bigger than RAM.

WHY THIS IS A SCRIPT AND NOT A RULE. The rule already existed (FINDINGS III.4
for speed tests, III.11a for smokes) and was violated THREE TIMES on 2026-08-21
by the same agent on the same box: a proposed d8 speed test, then a d8 smoke,
then a 143.682 GiB smoke on a 96 GiB machine that drove swap to 60 of 61 GiB.
A rule that gets restated after each violation is not a control. This is.

    ./preflight_ram.py <artifact-dir> [--headroom 0.90]

Exit 0 if the artifact's safetensors fit in physical RAM with headroom.
Exit 2 otherwise, naming the box, the artifact and where it CAN run.
Scoring (referee/score_streaming.py) is exempt: it streams by design.
"""
import argparse
import pathlib
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("artifact")
ap.add_argument("--headroom", type=float, default=0.90,
                help="fraction of physical RAM the artifact may occupy. mlx's "
                     "own recommended max is ~0.875 of RAM; 0.90 is already "
                     "generous and a smoke needs the KV cache on top.")
a = ap.parse_args()

P = pathlib.Path(a.artifact)
if not P.is_dir():
    sys.exit(f"PREFLIGHT FAIL: {P} is not a directory")

art = sum(f.stat().st_size for f in P.glob("*.safetensors"))
ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True).stdout.strip())
bar = ram * a.headroom
G = 2 ** 30
ok = art <= bar

print(f"PREFLIGHT  artifact {art/G:.3f} GiB   box RAM {ram/G:.0f} GiB   "
      f"bar {bar/G:.1f} GiB ({a.headroom:.0%})   -> {'OK' if ok else 'FAIL'}")
if not ok:
    print(f"PREFLIGHT FAIL: {P.name} needs {art/G:.3f} GiB resident and this "
          f"box has {ram/G:.0f} GiB. A generation smoke or speed test CANNOT "
          f"run here — it will thrash swap and produce no verdict.")
    print("  Run it on a box with more RAM. Scoring is exempt: the streaming "
          "referee streams by design.")
    sys.exit(2)
