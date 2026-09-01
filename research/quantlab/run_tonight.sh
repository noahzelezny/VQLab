#!/bin/bash
set -x
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
# RUN B — tail30 (debut candidate), 2-bit region = layers 0-29
./venv/bin/python vq_397b_fused.py --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail30" \
  --out "$M/zzvq-tail30-K1024" --vq-layers 0-29 --k 1024 || exit 1
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail30-K1024" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail30-K1024" --corpus referee/$c
done
# RUN A — tail3x3 (the 122G daily), 2-bit region = layers 0-56
./venv/bin/python vq_397b_fused.py --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$M/zzvq-tail3x3-K1024" --vq-layers 0-56 --k 1024 || exit 1
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K1024" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K1024" --corpus referee/$c
done
echo "ALL DONE"
