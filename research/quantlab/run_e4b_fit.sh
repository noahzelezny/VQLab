#!/bin/sh
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
SRC="/Volumes/Thunderbay SSD/Mlx_Models/hub/models--mlx-community--gemma-4-e4b-it-bf16/snapshots/eec12d0899edea9b738ab1009af9159cdfd70d71"
OUT="/Volumes/Thunderbay SSD/Exo Models/e4b-vq-d2K2048-mlp"
$V fit_e4b_vq.py --src "$SRC" --out "$OUT" 2>&1 | tee -a logs_live_$(basename run_e4b_fit.sh .sh).log | tail -5
echo "===== INDEPENDENT VERIFY ====="
$V verify_artifact.py --artifact "$OUT" --src "$SRC" --family gemma4_e4b --outlier 3.0 2>&1 | tee -a logs_live_$(basename run_e4b_fit.sh .sh).log | tail -8
