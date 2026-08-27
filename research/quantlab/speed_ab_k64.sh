#!/bin/sh
# A/B: new K64/K128 rung vs shipped VQ-2.2bpw. ONE PROCESS PER ARM (a single
# process holding both crashed the M4 at 118.6 GiB); each arm exits before the
# next loads. Two passes per arm, alternating, so drift shows up as scatter
# rather than as a fake winner.
set -e
cd "$(dirname "$0")"
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
NEW="$E/rotlab--397B-cheapshallow-k64-tail128-packed"
OLD="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw"
for pass in 1 2; do
  for arm in NEW OLD; do
    eval M=\$$arm
    echo "===== pass $pass  $arm  $(basename "$M")"
    $V m2_speed_split.py --model "$M" --contexts 512,2048 2>&1 | grep -viE "^loading|warn"
  done
done
echo "===== SPEED AB DONE"
