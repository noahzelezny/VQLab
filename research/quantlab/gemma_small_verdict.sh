#!/bin/sh
# E56 — gemma-small publish decision benches. Queued behind arm3/k8192.
# Pre-registered readings live in EXPERIMENTS.md E56; do not improvise new
# stopping rules after numbers land.
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SMALL="$E/gemma26b-rungs/vq-K256-d4"       # same artifact as the 79.81 run
E4B8="$E/mlx-community--gemma-4-e4b-it-8bit"
BF26=$(echo "/Volumes/Thunderbay SSD/Mlx_Models/hub/models--mlx-community--gemma-4-26b-a4b-it-bf16/snapshots"/*)

# Noah sequenced the 397B cheap-shallow build (vq_397b_fused) AHEAD of these
# benches. Wait for ALL fit/verify machinery, then hold a 20-min grace window
# so a job launched in the gap is not jumped.
while :; do
  until ! pgrep -f "vq_397b_codes|vq_397b_fused|pack_artifact|kl_ppl_calibrate|verify_artifact|score_streaming" >/dev/null; do sleep 60; done
  sleep 1200
  pgrep -f "vq_397b_codes|vq_397b_fused|pack_artifact|kl_ppl_calibrate|verify_artifact|score_streaming" >/dev/null || break
done

echo "== 1. litbench cyclic generative, per_item, gemma-small (rerun for pairing) =="
$V litbench_chat.py --model "$SMALL" --cyclic --generative \
  --out results_literary/gencyc_vq-K256-d4_v2.json 2>&1 | tee -a logs_live_$(basename gemma_small_verdict.sh .sh).log | tail -3
echo "== 2. litbench cyclic generative 26b bf16 (closes the non-cyclic hazard) =="
$V litbench_chat.py --model "$BF26" --cyclic --generative \
  --out results_literary/gencyc_26b-bf16.json 2>&1 | tee -a logs_live_$(basename gemma_small_verdict.sh .sh).log | tail -3
echo "== 3. paired McNemar: gemma-small vs e4b-8bit =="
$V paired_litbench.py results_literary/gencyc_vq-K256-d4_v2.json \
  results_literary/gencyc_e4b-8bit.json
echo "== 4. domain gens (instruct/summar/dialog), both arms =="
$V winrate_bench.py generate --model "$SMALL" --prompts winrate/prompts_domains.json \
  --max-tokens 640 --out winrate/gens_domains_small.json 2>&1 | tee -a logs_live_$(basename gemma_small_verdict.sh .sh).log | tail -1
$V winrate_bench.py generate --model "$E4B8" --prompts winrate/prompts_domains.json \
  --max-tokens 640 --out winrate/gens_domains_e4b8.json 2>&1 | tee -a logs_live_$(basename gemma_small_verdict.sh .sh).log | tail -1
echo "== 5. deterministic constraint pass-rate, paired =="
$V check_constraints.py --a winrate/gens_domains_small.json \
  --b winrate/gens_domains_e4b8.json
echo "########## E56 VERDICT BENCHES DONE — blind pairs + judging still owed"
