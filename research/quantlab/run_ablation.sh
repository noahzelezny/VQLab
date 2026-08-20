#!/bin/sh
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
echo "== ABLATION: where do the 20.8 mnats live? =="
echo "-- mlp-only (PLE stays 8-bit) --"
$V kl_damage.py score --model "$E/e4b-VQ-mlponly" --cache-dir "$E/kl_cache_e4b_LIT" 2>&1 | tail -2
echo "-- PLE-only (mlp stays 8-bit) --"
$V kl_damage.py score --model "$E/e4b-VQ-pleonly" --cache-dir "$E/kl_cache_e4b_LIT" 2>&1 | tail -2
echo "== ABLATION DONE (full=20.830, 8bit bar=8.149) =="
