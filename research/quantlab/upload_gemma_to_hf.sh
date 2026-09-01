#!/bin/bash
# Upload the two gemma artifacts to HF under TheDrainFlorist, cards + charts
# included. Publishing approved by Noah 2026-08-20. Plain arrays only —
# macOS bash 3.2 has no declare -A.
set -euo pipefail
export HF_HOME="${HF_HOME:-/Volumes/Thunderbay SSD/Mlx_Models}"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
E="/Volumes/Thunderbay SSD/Exo Models"
V=./venv/bin/python

upload_one() {
  name="$1"; dir="$2"; card="$3"; chart="$4"
  echo "===== $name"
  $V check_release.py --artifact "$dir" || exit 1
  printf '![chart](%s)\n\n' "$chart" | cat - "$card" > "$dir/README.md"
  cp "$chart" "$dir/"
  hf upload "TheDrainFlorist/$name" "$dir" . 2>&1 | tail -2
}

upload_one "gemma-4-26b-a4b-it-VQ-6.2bpw" \
  "$E/gemma26b-rungs/vq-K2048-d2-packed-sighted" \
  MODEL_CARD_GEMMA_QUALITY.md chart_gemma_ladder.png
upload_one "gemma-4-e4b-it-VQ-PLE" \
  "$E/e4b-VQ-pleonly-packed" \
  MODEL_CARD_GEMMA_E4B_VQPLE.md chart_e4b_vqple.png
echo "ALL UPLOADS DONE"
