#!/bin/bash
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
while pgrep -f k128_after >/dev/null; do sleep 120; done
# free the K1024 proxy (numbers recorded; rebuildable) before a 38h run
rm -rf "$M/zzvq-tail3x3-K1024"
echo "=== RUN G: tail3x3 + VQ d8 K16384 (2.00 bpw, ~100 GiB, ~38h) ==="
echo "=== THE ACCESSIBILITY ARTIFACT, premium geometry ==="
./venv/bin/python vq_397b_fused.py \
  --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$M/zzvq-tail3x3-d8K16384" --vq-layers 0-56 --dim 8 --k 16384 || exit 1
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-d8K16384" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail3x3-d8K16384" --corpus referee/$c
done
echo "G DONE"
