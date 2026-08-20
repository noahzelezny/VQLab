#!/bin/bash
# Upload the two gemma artifacts to HF under TheDrainFlorist, cards + charts
# included. Publishing approved by Noah 2026-08-20 ("publish the gemma models
# in the same collection with the charts and everything").
set -euo pipefail
export HF_HOME="${HF_HOME:-/Volumes/Thunderbay SSD/Mlx_Models}"
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
E="/Volumes/Thunderbay SSD/Exo Models"
V=./venv/bin/python

declare -A ART=(
  ["gemma-4-26b-a4b-it-VQ-6.2bpw"]="$E/gemma26b-rungs/vq-K2048-d2-packed-sighted|MODEL_CARD_GEMMA_QUALITY.md|chart_gemma_ladder.png"
  ["gemma-4-e4b-it-VQ-PLE"]="$E/e4b-VQ-pleonly-packed|MODEL_CARD_GEMMA_E4B_VQPLE.md|chart_e4b_vqple.png"
)
for name in "${!ART[@]}"; do
  IFS='|' read -r dir card chart <<< "${ART[$name]}"
  echo "===== $name"
  $V check_release.py --artifact "$dir" || exit 1
  # stage card as README + chart into the artifact dir
  printf '![chart](%s)\n\n' "$chart" | cat - "$card" > "$dir/README.md"
  cp "$chart" "$dir/"
  HF_HOME="$HF_HOME" hf upload "TheDrainFlorist/$name" "$dir" . 2>&1 | tail -2
done
echo "ALL UPLOADS DONE"
