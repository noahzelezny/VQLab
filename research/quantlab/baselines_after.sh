#!/bin/bash
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
# wait for the main chain to finish A (no GPU overlap with a fit — rule 6)
while pgrep -f m3_followon.sh >/dev/null; do sleep 120; done
echo "=== MISSING BASELINES for the 2.6bit-class claim ==="
for m in "TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
         "spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit"; do
  for c in referee_corpus.txt referee_corpus_code.txt; do
    ./venv/bin/python referee/score_streaming.py --model "$M/$m" --corpus referee/$c
    ./venv/bin/python referee/score_streaming.py --model "$M/$m" --corpus referee/$c
  done
done
echo "BASELINES DONE"
