#!/bin/bash
# Upload the three VQ artifacts to HuggingFace under TheDrainFlorist.
# NOT run automatically — publishing is Noah's call, under Noah's account.
#
#   huggingface-cli login          # once, if not already
#   ./upload_to_hf.sh --dry-run    # list what would go
#   ./upload_to_hf.sh 2.2          # upload one
#   ./upload_to_hf.sh all
set -euo pipefail
E="/Volumes/Thunderbay SSD/Exo Models"
ORG="TheDrainFlorist"
declare -a ARTS=("Qwen3.5-397B-A17B-VQ-2.2bpw" "Qwen3.5-397B-A17B-VQ-2.4bpw" "Qwen3.5-397B-A17B-VQ-3.1bpw")

sel="${1:-all}"
for a in "${ARTS[@]}"; do
  case "$sel" in
    all|--dry-run) ;;
    *) [[ "$a" == *"$sel"* ]] || continue ;;
  esac
  size=$(du -sh "$E/$a" | cut -f1)
  echo "=== $ORG/$a  ($size) ==="
  # sanity gates before anything leaves the machine
  [ -f "$E/$a/README.md" ] || { echo "  !! no model card"; exit 1; }
  [ -f "$E/$a/model.py" ] || { echo "  !! no model.py (downloaders could not load it)"; exit 1; }
  grep -q "MUST be derived from the CURRENT tensors" "$E/$a/model.py" \
    || { echo "  !! model.py predates the sharding fix — rerun add_model_file.py"; exit 1; }
  python3 -c "
import json,sys
c=json.load(open('$E/$a/config.json'))
assert c.get('model_file')=='model.py', 'config missing model_file'
assert c.get('vq_modules'), 'config missing vq_modules'
print('  config ok:', len(c['vq_modules']), 'vq modules')"
  if [ "$sel" = "--dry-run" ]; then echo "  (dry run — nothing uploaded)"; continue; fi
  huggingface-cli upload "$ORG/$a" "$E/$a" . --repo-type model
done
