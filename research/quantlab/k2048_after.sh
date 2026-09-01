#!/bin/bash
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
while pgrep -f k512_after >/dev/null; do sleep 120; done
# free the K256 proxy first — its numbers are recorded, rebuildable in ~5h
rm -rf "$M/zzvq-tail3x3-K256"
echo "=== RUN E: tail3x3 + VQ K2048 (3.00 bpw, ~145 GiB) ==="
echo "=== tests the AXIS question: codebook size vs tail depth ==="
./venv/bin/python vq_397b_fused.py \
  --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$M/zzvq-tail3x3-K2048" --vq-layers 0-56 --k 2048 || exit 1
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K2048" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-K2048" --corpus referee/$c
done
echo "E DONE"
