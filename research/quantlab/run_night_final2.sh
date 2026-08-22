#!/bin/sh
# NIGHT SCHEDULE v2 (M3), strictly sequential. Replaces run_night_final.sh,
# whose watcher was stopped (no compute lost — E121 runs in its own process).
# Change: E124 is d2/K256, not d2/K512. The K512 variant lands at 14.59 GiB,
# ABOVE q4's 14.094 — my 13.1 GiB estimate used a non-MLP figure I invented
# rather than measured. Verified against our own artifact: predicted 9.611 vs
# measured 9.612 GiB for the d4/K256 build. K256 lands at 13.594, inside the
# band Noah asked for, and is the LARGEST d2 rung that fits under q4.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
Q=logs_night_final.log
say() { echo "$(date '+%H:%M:%S')  $*" >> $Q; }

while pgrep -f fitter_0816_cdcdeab >/dev/null; do sleep 60; done
say "E121 fit finished"

C=$(grep -c "relerr 1.0000" logs_live_e121_oldfitter.log)
say "E121 collapse scan: $C tensors at 1.0000"
if [ "$C" -gt 0 ]; then
  say "E121 CONTAMINATED — not gating, not scoring, not comparing. A refit is a re-roll, not a fix."
else
  say "E121 clean — gate/pack/score"
  sh run_e121_oldfitter.sh >> logs_e121_tail.log 2>&1
  say "E121 chain exit=$?"
fi

say "E120 start"
$V probe_accum_order.py --ks 256,2048 > logs_live_e120_accum.log 2>&1
say "E120 exit=$?"

say "E124 start (d2 K256, target band under q4)"
sh run_e124_27b_d2K256.sh >> logs_e124_chain.log 2>&1
say "E124 exit=$?"

say "M3 NIGHT SCHEDULE COMPLETE"
