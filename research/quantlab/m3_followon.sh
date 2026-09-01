#!/bin/bash
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
score () { for c in referee_corpus.txt referee_corpus_code.txt; do
    ./venv/bin/python referee/score_streaming.py --model "$1" --corpus referee/$c
    ./venv/bin/python referee/score_streaming.py --model "$1" --corpus referee/$c
  done; }
while kill -0 $(cat /tmp/bpid) 2>/dev/null; do sleep 60; done
echo "=== B DONE — scoring tail30+VQ K1024 (baseline 2.3982 / 2.5928) ==="
score "$M/zzvq-tail30-K1024"
echo "=== ALLOCATION PROBE ==="
./venv/bin/python probe_vq_alloc.py
echo "=== RUN C: tail3x3 + VQ K256 (2.25 bpw, smaller than RTN) ==="
./venv/bin/python vq_397b_fused.py \
  --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$M/zzvq-tail3x3-K256" --vq-layers 0-56 --k 256 || exit 1
score "$M/zzvq-tail3x3-K256"
echo "=== RUN A: tail3x3 + VQ K1024 (the undiluted case) ==="
./venv/bin/python vq_397b_fused.py \
  --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$M/zzvq-tail3x3-K1024" --vq-layers 0-56 --k 1024 || exit 1
echo "=== scoring tail3x3+VQ K1024 (baseline 3.1557) ==="
score "$M/zzvq-tail3x3-K1024"
echo "ALL DONE"
