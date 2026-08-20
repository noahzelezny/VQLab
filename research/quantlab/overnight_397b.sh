#!/bin/sh
# OVERNIGHT (Noah, 08-19 ~21:45): after the gemma PLE-only gates finish —
#   1. rebuild the deleted 397B struct6-tail3x3 base (bf16 src survives)
#   2. run the 397B cheap-shallow fit (peer session's recipe, Noah-approved)
#   3. verify + 6-bit pack smoke
#   4. start the scout nightly-dispatcher watchdog
# Recipe details live in overnight_recipe.env so a peer correction can be
# dropped in WITHOUT editing this script while it waits.
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
. ./overnight_recipe.env

echo "=== waiting for gemma PLE-only gates ==="
while pgrep -f "run_pleonly_gates|litbench_chat" >/dev/null; do sleep 120; done
echo "=== gemma gates done at $(date -Iseconds); final lines: ==="
tail -6 logs_pleonly_gates.log

echo "=== 1. rebuild base: $REBUILD_NAME ==="
$V convert_variant.py --name "$REBUILD_NAME" $REBUILD_FLAGS 2>&1 | tee -a logs_live_397b.log | tail -4 || exit 1

BASE="$E/TheDrainFlorist--Qwen3.5-397B-A17B-$REBUILD_NAME"
echo "=== 2. cheap-shallow fit ==="
$V $FIT_TOOL --family qwen3_5 --base "$BASE" --src "$E/Qwen--Qwen3.5-397B-A17B-bf16/" \
   --out "$E/vq397-cheapshallow-k64-tail256" $FIT_FLAGS 2>&1 | tee -a logs_live_397b.log | tail -3 || exit 1

echo "=== 3. verify (read per-region: mixed geometry, E57 amendment) ==="
$V verify_artifact.py --artifact "$E/vq397-cheapshallow-k64-tail256" \
   --src "$E/Qwen--Qwen3.5-397B-A17B-bf16" --family qwen3_5 --outlier 3.0 2>&1 | tee -a logs_live_397b.log | tail -6

echo "=== 4. scout dispatcher watchdog ==="
cd /Users/noahzelezny/Documents/AgenicAI
STOP=0; FAILS=0
nohup caffeinate -i bash -c 'while true; do echo "[watchdog] start $(date -Iseconds)"; ( exec -a nightly-dispatcher .venv/bin/python -m scout.services.nightly_dispatcher --node m3 --job-timeout 21600 ); ec=$?; echo "[watchdog] exited $ec, restart in 5s"; sleep 5; done' > /tmp/scout_dispatcher_0820.log 2>&1 &
echo "dispatcher watchdog up ($!)"
echo "########## OVERNIGHT CHAIN DONE $(date -Iseconds)"
