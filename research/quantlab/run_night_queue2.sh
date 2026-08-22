#!/bin/sh
# NIGHT QUEUE 2 (08-21, Noah walking): after E118 clears, answer HIS question —
# how did the shipped 2.4 happen? E120 (cheap probe) then E121 (the actual
# 08-16 fitter). Neither launches if E118's chain failed.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
L118=logs_live_e116_crossover.log
while :; do
  grep -q "E118 DONE" $L118 2>/dev/null && break
  if grep -qE "FIT FAILED|ABORT:" $L118 2>/dev/null; then
    echo "E118 FAILED — E120/E121 not launched" >> logs_night_queue2.log; exit 1
  fi
  sleep 120
done
echo "E118 done $(date '+%H:%M:%S') — E120 probe" >> logs_night_queue2.log
./venv/bin/python probe_accum_order.py --ks 256,2048 \
    > logs_live_e120_accum.log 2>&1
echo "E120 done $(date '+%H:%M:%S') — E121 old fitter" >> logs_night_queue2.log
exec sh run_e121_oldfitter.sh
