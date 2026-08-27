#!/bin/sh
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
while pgrep -f "litbench_chat" >/dev/null; do sleep 60; done
echo "== 3-retry. KL-to-bf16: 8-bit incumbent (--allow-unmatched: dead shared-KV tensors, see kl_damage E-SERIES NOTE) =="
$V kl_damage.py score --model "$E/mlx-community--gemma-4-e4b-it-8bit" \
   --cache-dir "$E/kl_cache_e4b_LIT" --allow-unmatched 2>&1 | tee -a logs_live_e4b_benches.log | tail -3
echo "== 3b. KL-to-bf16: PACKED VQ artifact (identity check at the KL level) =="
$V kl_damage.py score --model "$E/e4b-VQ-d2K2048-packed" \
   --cache-dir "$E/kl_cache_e4b_LIT" 2>&1 | tee -a logs_live_e4b_benches.log | tail -3
echo "== 5. paired McNemar: VQ vs 8-bit incumbent =="
$V paired_litbench.py results_literary/gencyc_e4b-VQ.json results_literary/gencyc_e4b-8bit.json
echo "########## E68 BENCHES COMPLETE"
