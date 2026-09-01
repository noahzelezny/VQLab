#!/bin/sh
# QUEUE 3: E118 chain -> E120 probe -> PUBLISH VERIFY -> E121 (08-16 fitter).
# Publish verification moved ahead of E121 on Noah's call: E121 is ~1hr, the
# verification is ~15 min and gates a decision he is actively making.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
L118=logs_live_e116_crossover.log
while :; do
  grep -q "E118 DONE" $L118 2>/dev/null && break
  if grep -qE "FIT FAILED|ABORT:" $L118 2>/dev/null; then
    echo "E118 FAILED — nothing downstream launched" >> logs_night_queue3.log; exit 1
  fi
  sleep 60
done
echo "E118 done $(date '+%H:%M:%S') — E120 probe" >> logs_night_queue3.log
./venv/bin/python probe_accum_order.py --ks 256,2048 > logs_live_e120_accum.log 2>&1
echo "E120 done $(date '+%H:%M:%S') — publish verify" >> logs_night_queue3.log
sh run_publish_verify.sh >> logs_night_queue3.log 2>&1
echo "publish verify done $(date '+%H:%M:%S') — E121 old fitter" >> logs_night_queue3.log
exec sh run_e121_oldfitter.sh
