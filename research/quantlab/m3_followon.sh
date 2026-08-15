#!/bin/bash
M="/Volumes/Thunderbay SSD/Exo Models"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
while kill -0 $(cat /tmp/bpid) 2>/dev/null; do sleep 60; done
echo "=== B FIT DONE — scoring tail30+VQ (baseline 2.3982 / 2.5928) ==="
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail30-K1024" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$M/zzvq-tail30-K1024" --corpus referee/$c
done
echo "=== ALLOCATION PROBE ==="
./venv/bin/python probe_vq_alloc.py
# M4 abandoned tonight (T7 reads zeros + Metal timeouts on BOTH mlx builds).
# here (single-box instrument of record). Fallback: if the M4 fit dies and
# no index appears within 6h, fit A locally.
echo "=== RUN A on M3 ==="
A="$M/zzvq-tail3x3-K1024"
./venv/bin/python vq_397b_fused.py \
  --base "$M/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3" \
  --out "$A" --vq-layers 0-56 --k 1024 || exit 1
echo "=== scoring tail3x3+VQ (baseline 3.1557) ==="
for c in referee_corpus.txt referee_corpus_code.txt; do
  ./venv/bin/python referee/score_streaming.py --model "$A" --corpus referee/$c
  ./venv/bin/python referee/score_streaming.py --model "$A" --corpus referee/$c
done
echo "ALL DONE"
