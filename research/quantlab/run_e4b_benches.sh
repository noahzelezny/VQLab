#!/bin/sh
# E68: the full measurement battery for the VQ e4b vs the 8-bit incumbent.
# gemma raw ppl is INVALID (model property) — the instruments are:
#   1. KL-to-bf16 for BOTH artifacts (the "is it better than 8bit" number)
#   2. litbench cyclic generative + per-item -> paired McNemar vs 8-bit
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
BF16="/Volumes/Thunderbay SSD/Mlx_Models/hub/models--mlx-community--gemma-4-e4b-it-bf16/snapshots/eec12d0899edea9b738ab1009af9159cdfd70d71"
VQ="$E/e4b-VQ-d2K2048"
Q8="$E/mlx-community--gemma-4-e4b-it-8bit"
CACHE="$E/kl_cache_e4b_LIT"

echo "== 1. build e4b bf16 teacher cache (literary corpus) =="
$V kl_damage.py cache --model "$BF16" --out-dir "$CACHE" \
   --corpus referee/referee_corpus_literary.txt 2>&1 | tee -a logs_live_e4b_benches.log | tail -2
echo "== 2. KL-to-bf16: VQ artifact =="
$V kl_damage.py score --model "$VQ" --cache-dir "$CACHE" 2>&1 | tee -a logs_live_e4b_benches.log | tail -3
echo "== 3. KL-to-bf16: 8-bit incumbent (the bar) =="
$V kl_damage.py score --model "$Q8" --cache-dir "$CACHE" 2>&1 | tee -a logs_live_e4b_benches.log | tail -3
echo "== 4. litbench cyclic generative, VQ e4b =="
$V litbench_chat.py --model "$VQ" --cyclic --generative \
   --out results_literary/gencyc_e4b-VQ.json 2>&1 | tee -a logs_live_e4b_benches.log | tail -2
echo "== 5. paired McNemar: VQ vs 8-bit incumbent =="
$V paired_litbench.py results_literary/gencyc_e4b-VQ.json results_literary/gencyc_e4b-8bit.json
echo "########## E68 BENCHES DONE"
