#!/bin/sh
# NIGHT QUEUE (08-21, Noah asleep): launch E116 when E115's chain completes.
# Watches for the E115 DONE banner (or FAILED/ABORT) in its live log.
# On E115 failure it does NOT launch E116 — a crossover fit downstream of a
# broken chain is machine time spent making an uninterpretable number.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
L115=logs_live_e115_randinit.log
while :; do
  grep -q "E115 DONE" $L115 2>/dev/null && break
  if grep -qE "FIT FAILED|ABORT:" $L115 2>/dev/null; then
    echo "E115 FAILED — E116 not launched" >> logs_night_queue.log; exit 1
  fi
  sleep 120
done
echo "E115 done at $(date '+%H:%M:%S') — launching E116" >> logs_night_queue.log
exec sh run_e116_crossover.sh
