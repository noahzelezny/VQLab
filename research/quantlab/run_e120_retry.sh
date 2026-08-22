#!/bin/sh
# E120 retry — the first attempt died in 1 second on a KeyError (it guessed
# the ARTIFACT key convention for a SOURCE read). No data was produced and
# none was lost. Waits for the publish verification to clear so a 100 GiB
# load and this probe are not contending.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
while pgrep -f "run_publish_verify|score_streaming" >/dev/null; do sleep 30; done
./venv/bin/python probe_accum_order.py --ks 256,2048 > logs_live_e120_accum.log 2>&1
echo "E120 retry done $(date '+%H:%M:%S')" >> logs_night_queue3.log
