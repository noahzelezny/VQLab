#!/bin/bash
# Pack E and F to their headline sizes, then referee the packed copies.
# Packed scores MUST equal the unpacked ones exactly (representation change).
set -e
cd "$(dirname "$0")"
PY=venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
for n in "rotlab--397B-tail3x3-vqK2048codes" "rotlab--397B-tail3x3-vqK128codes"; do
  echo "=== PACK $n $(date) ==="
  rm -rf "$E/$n-packed"
  $PY pack_artifact.py --src "$E/$n" --out "$E/$n-packed"
  for c in referee/referee_corpus.txt referee/referee_corpus_code.txt; do
    echo "=== REFEREE PACKED $n $c ==="
    for r in 1 2; do $PY referee/score_streaming.py --model "$E/$n-packed" --corpus "$c"; done
  done
  du -sh "$E/$n" "$E/$n-packed"
done
echo "PACK E+F COMPLETE $(date)"
