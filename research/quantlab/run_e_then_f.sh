#!/bin/bash
# E (d4 K2048) then F (d4 K128) codes fits + referee x2 both corpora each.
# M3-local: base/src/out all on the Thunderbay SSD (no SMB staging needed).
set -e
cd "$(dirname "$0")"
PY=venv/bin/python
EXO="/Volumes/Thunderbay SSD/Exo Models"
BASE="$EXO/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3"
SRC="$EXO/Qwen--Qwen3.5-397B-A17B-bf16"

for spec in "2048:rotlab--397B-tail3x3-vqK2048codes" \
            "128:rotlab--397B-tail3x3-vqK128codes"; do
  K="${spec%%:*}"; NAME="${spec#*:}"; OUT="$EXO/$NAME"
  echo "=== FIT K=$K -> $NAME  $(date) ==="
  $PY vq_397b_codes.py --base "$BASE" --src "$SRC" --out "$OUT" \
      --vq-layers 0-56 --k "$K" --dim 4
  $PY add_model_file.py --artifact "$OUT" --k "$K" --dim 4
  for corpus in referee/referee_corpus.txt referee/referee_corpus_code.txt; do
    echo "=== REFEREE $NAME $corpus ==="
    for run in 1 2; do
      $PY referee/score_streaming.py --model "$OUT" --corpus "$corpus"
    done
  done
done
echo "E+F CHAIN COMPLETE $(date)"
