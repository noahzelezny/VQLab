#!/bin/sh
# Recovery relaunch. E120 and E121 both died at 21:43 — not from their own
# defects but because the box was still thrashing from an oversized smoke
# (swap 78 GiB of 79 GiB at launch). Swap is back to normal. E120 first
# (minutes), then E121 (~1hr); NEVER concurrently, which is half of why they
# died. E121 RESUMES — shard 1 already exists from a 90s diagnostic run and
# will be skipped, which is correct behaviour, not a stale-shard risk.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
echo "recover start $(date '+%H:%M:%S')" > logs_recover.log

$V probe_accum_order.py --ks 256,2048 > logs_live_e120_accum.log 2>&1
echo "E120 exit=$? $(date '+%H:%M:%S')" >> logs_recover.log

exec sh run_e121_oldfitter.sh
