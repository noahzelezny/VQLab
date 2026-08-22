#!/bin/sh
# NIGHT SCHEDULE, M3. Strictly sequential — concurrency killed two experiments
# tonight. Each stage gated; a failure stops the chain rather than cascading.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
Q=logs_night_final.log
say() { echo "$(date '+%H:%M:%S')  $*" >> $Q; }

# 1. wait out E121 (already running)
while pgrep -f fitter_0816_cdcdeab >/dev/null; do sleep 60; done
say "E121 fit finished"

# 2. E121 COLLAPSE SCAN before anything downstream trusts it (M4's protocol:
#    the 08-16 fitter has no abort, so its log is the compute-side catcher)
C=$(grep -c "relerr 1.0000" logs_live_e121_oldfitter.log)
say "E121 collapse scan: $C tensors at 1.0000"
if [ "$C" -gt 0 ]; then
  say "E121 CONTAMINATED — not gating, not scoring, not comparing. Refit is a re-roll."
else
  say "E121 clean — gate/pack/score"
  sh run_e121_oldfitter.sh >> logs_e121_tail.log 2>&1
  say "E121 chain done"
fi

# 3. E120 accumulation probe (fixed + smoke-tested on a 1-expert case)
say "E120 start"
$V probe_accum_order.py --ks 256,2048 > logs_live_e120_accum.log 2>&1
say "E120 exit=$?"

# 4. E124 — Noah's target: 27B at 4-bit size via d2
say "E124 start"
sh run_e124_27b_d2.sh >> logs_e124_chain.log 2>&1
say "E124 exit=$?"

say "M3 NIGHT SCHEDULE COMPLETE"
