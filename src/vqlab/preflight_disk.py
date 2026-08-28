#!/usr/bin/env python3
"""HARD GUARD: refuse a build whose output volume lacks the space to hold it.

Sibling of preflight_ram.py, same contract. A fit/pack/convert that runs out
of disk mid-write leaves a half-written artifact that LOOKS like a directory
of safetensors (index present, shards truncated) — and on a boot volume it
can wedge the whole box (macOS degrades hard near 0 bytes free). The check
is one statvfs; there is no reason to ever skip it.

    ./preflight_disk.py --out <dir-or-parent> --need-gib 96 [--headroom 1.15]
    ./preflight_disk.py --out <dir> --like <existing-artifact-dir>

--need-gib states the expected artifact size (use `vqlab price` output);
--like sizes it from an existing artifact's safetensors instead (e.g. the
previous rung of the same ladder). --headroom (default 1.15) covers shard
staging and index rewrites. Boot volumes get an EXTRA absolute floor of
20 GiB free-after-write, because "fits exactly" on the boot disk is how a
box bricks.

Exit 0 = fits. Exit 2 = does not, naming the shortfall.
"""
import argparse
import os
import pathlib
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True,
                help="output dir (or its intended parent, if not yet created)")
ap.add_argument("--need-gib", type=float, default=None)
ap.add_argument("--like", default=None,
                help="size the need from this artifact's *.safetensors")
ap.add_argument("--headroom", type=float, default=1.15)
a = ap.parse_args()
if (a.need_gib is None) == (a.like is None):
    sys.exit("PREFLIGHT FAIL: give exactly one of --need-gib / --like")

G = 2 ** 30
if a.like:
    need = sum(f.stat().st_size for f in pathlib.Path(a.like).glob("*.safetensors"))
    if need == 0:
        sys.exit(f"PREFLIGHT FAIL: --like {a.like} has no safetensors to size from")
else:
    need = int(a.need_gib * G)
need = int(need * a.headroom)

out = pathlib.Path(a.out)
probe = out
while not probe.exists():
    probe = probe.parent
st = os.statvfs(probe)
free = st.f_bavail * st.f_frsize

boot = os.stat(probe).st_dev == os.stat("/").st_dev
floor = 20 * G if boot else 0
ok = free - need >= floor

print(f"PREFLIGHT  need {need/G:.1f} GiB (incl. x{a.headroom} headroom)   "
      f"free {free/G:.1f} GiB on {probe}"
      + (f"   [boot volume: +{floor/G:.0f} GiB floor]" if boot else ""))
if ok:
    print("OK: fits")
    sys.exit(0)
print(f"FAIL: short by {(need + floor - free)/G:.1f} GiB. Free space or "
      f"choose another volume — a mid-write ENOSPC leaves a truncated "
      f"artifact that still looks like one.")
sys.exit(2)
