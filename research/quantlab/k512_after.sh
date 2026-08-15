#!/bin/bash
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
while pgrep -f baselines_after >/dev/null; do sleep 120; done
echo "=== RUN D: tail3x3 + VQ K512 (2.50 bpw = RTN's budget, ~122 GiB) ==="
./venv/bin/python vq_397b_fused.py \
  --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$M/zzvq-tail3x3-K512" --vq-layers 0-56 --k 512 || exit 1
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K512" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K512" --corpus referee/$c
done
echo "D DONE"
