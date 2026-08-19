#!/bin/sh
# E63 capacity probe: gemma-small (VQ 2.25bpw of 26b) vs e4b-8bit.
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
$V winrate_bench.py generate --model "$E/gemma26b-rungs/vq-K256-d4" \
  --prompts winrate/prompts_capacity.json --max-tokens 1400 \
  --out winrate/gens_capacity_small.json 2>&1 | tail -1
$V winrate_bench.py generate --model "$E/mlx-community--gemma-4-e4b-it-8bit" \
  --prompts winrate/prompts_capacity.json --max-tokens 1400 --allow-unmatched \
  --out winrate/gens_capacity_e4b8.json 2>&1 | tail -1
echo "===== PAIRED CAPACITY SCORES ====="
$V score_capacity.py --a winrate/gens_capacity_small.json \
  --b winrate/gens_capacity_e4b8.json
