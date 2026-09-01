#!/bin/sh
# Cheap-shallow the whole 397B ladder (Noah, 08-20): 2.2-class and 3.1-class
# counterparts of the shipped flat rungs. Same base (struct6-tail3x3), same
# mechanism as the proven 2.3 (quarter codebook on L0-9, full on body).
# ONE sequential script; all values INLINE (the env-file indirection is what
# scattered last night's artifact); tee everywhere; never edited while running.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
BASE="$E/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3"
SRC="$E/Qwen--Qwen3.5-397B-A17B-bf16"
SHIPPED="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
L=logs_live_ladder397b.log

run_rung() {
  out="$1"; k="$2"; tailgeom="$3"; abort="$4"
  echo "########## RUNG $out (shallow K$k / body $tailgeom)" | tee -a $L
  $V vq_397b_codes.py --family qwen3_5 --base "$BASE" --src "$SRC" \
     --out "$E/$out" --vq-layers 0-56 --k "$k" --dim 4 \
     --tail-from 10 --tail-geom "$tailgeom" --relerr-abort "$abort" \
     --expert-chunk 8 2>&1 | tee -a $L | tail -2 || { echo "FIT FAILED $out" | tee -a $L; return 1; }
  $V add_model_file.py --artifact "$E/$out" >/dev/null 2>&1
  cp "$SHIPPED/tokenizer.json" "$SHIPPED/tokenizer_config.json" "$E/$out/"
  $V verify_artifact.py --artifact "$E/$out" --src "$SRC" --family qwen3_5 \
     --outlier 3.0 2>&1 | tee -a $L | tail -4
  $V pack_artifact.py --src "$E/$out" --out "$E/$out-packed" 2>&1 | tee -a $L | tail -1
  cp "$SHIPPED/tokenizer.json" "$SHIPPED/tokenizer_config.json" "$E/$out-packed/"
  $V graft_vision.py --artifact "$E/$out-packed" --src "$SRC" \
     --prefixes model.visual 2>&1 | tee -a $L | tail -1
  $V check_vision.py --artifact "$E/$out-packed" --src "$SRC" 2>&1 | tee -a $L | tail -1
  $V check_release.py --artifact "$E/$out-packed" 2>&1 | tee -a $L | tail -1
  $V referee/score_streaming.py --model "$E/$out-packed" --corpus referee/referee_corpus.txt 2>&1 | tee -a $L | tail -1
  $V referee/score_streaming.py --model "$E/$out-packed" --corpus referee/referee_corpus_code.txt 2>&1 | tee -a $L | tail -1
}

run_rung "rotlab--397B-cheapshallow-k32-tail128"  32  d4k128  0.75
run_rung "rotlab--397B-cheapshallow-k512-tail2048" 512 d4k2048 0.70
echo "########## LADDER DONE — refs: shipped 2.2 ppl 3.1706 @100.9G, shipped 3.1 @143.7G (referee ref TBD), cheapshallow-2.3 2.779/2.6479 @107.9G" | tee -a $L
