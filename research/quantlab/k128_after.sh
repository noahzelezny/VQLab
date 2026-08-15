#!/bin/bash
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
while pgrep -f k2048_after >/dev/null; do sleep 120; done
echo "=== RUN F: tail3x3 + VQ d4 K128 (2.00 bpw, ~100 GiB) ==="
echo "=== THE ACCESSIBILITY ARTIFACT: fits a 128 GB machine ==="
./venv/bin/python vq_397b_fused.py \
  --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$M/zzvq-tail3x3-K128" --vq-layers 0-56 --k 128 || exit 1
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K128" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K128" --corpus referee/$c
done
echo "F DONE"
