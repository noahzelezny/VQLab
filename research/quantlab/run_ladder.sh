#!/bin/sh
# E64: escalating ladder — find each model's breaking point.
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
for pair in "small:$E/gemma26b-rungs/vq-K256-d4:" \
            "e4b8:$E/mlx-community--gemma-4-e4b-it-8bit:--allow-unmatched"; do
  tag=$(echo "$pair" | cut -d: -f1); mdl=$(echo "$pair" | cut -d: -f2); flag=$(echo "$pair" | cut -d: -f3)
  echo "########## GENERATING: $tag"
  $V winrate_bench.py generate --model "$mdl" --prompts winrate/prompts_ladder.json \
     --max-tokens 4600 $flag --out winrate/gens_ladder_$tag.json 2>&1 | tee -a logs_live_$(basename run_ladder.sh .sh).log | tail -1
  echo "---------- partial scores: $tag"
  $V score_ladder.py --gens winrate/gens_ladder_$tag.json
done
echo "===== PAIRED LADDER RESULT ====="
$V score_ladder.py --a winrate/gens_ladder_small.json --b winrate/gens_ladder_e4b8.json
