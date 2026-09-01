#!/bin/sh
# A/B: d8-K16384 (14-bit codes, NO u8view path) vs shipped VQ-2.2bpw.
# Size-matched to within 70 MiB, so a throughput gap is code-width, not bytes.
# MUST run on M4: each arm is ~101 GiB and M3 has 96 GiB (rule III.4). ONE PROCESS PER ARM (a single
# process holding both crashed the M4 at 118.6 GiB); each arm exits before the
# next loads. Two passes per arm, alternating, so drift shows up as scatter
# rather than as a fake winner.
set -e
cd "$(dirname "$0")"
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
NEW="$E/rotlab--397B-d8K16384-packed"
OLD="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw"
for pass in 1 2; do
  for arm in NEW OLD; do
    eval M=\$$arm
    echo "===== pass $pass  $arm  $(basename "$M")"
    $V m2_speed_split.py --model "$M" --contexts 512,2048 2>&1 | grep -viE "^loading|warn"
  done
done
echo "===== SPEED AB DONE"
