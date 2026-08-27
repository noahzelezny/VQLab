#!/bin/sh
# E80 gate chain: the M4-fitted 3.1-class rung, gated on M3 per the split.
set -e
cd "$(dirname "$0")"
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.5-397B-A17B-bf16"
out="rotlab--397B-cheapshallow-k512-tail2048"
L=logs_e80_chain.log
$V add_model_file.py --artifact "$E/$out" >/dev/null 2>&1
$V verify_artifact.py --artifact "$E/$out" --src "$SRC" --family qwen3_5 --outlier 3.0 2>&1 | tail -4
$V pack_artifact.py --src "$E/$out" --out "$E/$out-packed" 2>&1 | tail -1
cp "$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw/tokenizer.json" "$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw/tokenizer_config.json" "$E/$out-packed/" 2>/dev/null || true
$V graft_vision.py --artifact "$E/$out-packed" --src "$SRC" --prefixes model.visual 2>&1 | tail -1
$V check_vision.py --artifact "$E/$out-packed" --src "$SRC" 2>&1 | tail -1
$V check_release.py --artifact "$E/$out-packed" 2>&1 | tail -1
$V check_bundle.py --artifact "$E/$out-packed" 2>&1 | tail -1
$V referee/score_streaming.py --model "$E/$out-packed" --corpus referee/referee_corpus.txt 2>&1 | tail -1
$V referee/score_streaming.py --model "$E/$out-packed" --corpus referee/referee_corpus_code.txt 2>&1 | tail -1
echo "########## E80 CHAIN DONE"
