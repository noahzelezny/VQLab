#!/bin/bash
# Upload the three VQ artifacts to HuggingFace under TheDrainFlorist.
# NOT run automatically — publishing is Noah's call, under Noah's account.
#
#   hf auth login                  # once, if not already (huggingface-cli is deprecated)
#   ./upload_to_hf.sh --dry-run    # list what would go
#   ./upload_to_hf.sh 2.2          # upload one
#   ./upload_to_hf.sh all
set -euo pipefail
# Noah's HF_HOME lives on the SSD (~/.zshrc), and the auth token with it.
# Non-interactive shells do NOT source .zshrc, so without this every hf call
# runs UNAUTHENTICATED and a 100+ GiB push fails at the end. Set it explicitly.
export HF_HOME="${HF_HOME:-/Volumes/Thunderbay SSD/Mlx_Models}"

E="/Volumes/Thunderbay SSD/Exo Models"
ORG="TheDrainFlorist"
declare -a ARTS=("Qwen3.5-397B-A17B-VQ-2.2bpw" "Qwen3.5-397B-A17B-VQ-2.4bpw" "Qwen3.5-397B-A17B-VQ-3.1bpw")

python3 -c "
from huggingface_hub import HfApi
w=HfApi().whoami(); r=(w.get('auth') or {}).get('accessToken',{}).get('role')
print(f'authenticated as {w[\"name\"]} (token role: {r})')
assert r in ('write','admin'), 'token lacks write permission — uploads would fail'
" || { echo 'HF auth pre-flight FAILED'; exit 1; }

sel="${1:-all}"
for a in "${ARTS[@]}"; do
  case "$sel" in
    all|--dry-run) ;;
    *) [[ "$a" == *"$sel"* ]] || continue ;;
  esac
  size=$(du -sh "$E/$a" | cut -f1)
  echo "=== $ORG/$a  ($size) ==="
  # sanity gates before anything leaves the machine
  # __pycache__ is build residue from importing model.py locally — never publish it
  rm -rf "$E/$a/__pycache__" 2>/dev/null
  find "$E/$a" -name ".DS_Store" -delete 2>/dev/null   # macOS junk; it WILL get published otherwise
  for need in tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors.index.json; do
    [ -f "$E/$a/$need" ] || { echo "  !! missing $need — downloaders could not run it"; exit 1; }
  done
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
  # PRIVATE first by default: inspect the rendered card + file listing on the
  # Hub before it is visible to anyone. Flip with `hf repo settings` (or the
  # web UI) once it looks right. Pass PUBLIC=1 to skip the private stage.
  # PUBLIC by default: a free account's PRIVATE storage quota is far below
  # 100 GiB, so private staging fails at COMMIT time with
  # "403 Private repository storage limit reached" after the data is already
  # uploaded (learned on 2.2bpw, 2026-08-16). Public repos have no such cap.
  # Set PRIVATE=1 only if the account has paid private storage.
  vis=""; [ "${PRIVATE:-0}" = "1" ] && vis="--private"
  hf repo create "$ORG/$a" --repo-type model $vis --exist-ok 2>&1 | tail -1
  # upload-large-folder, NOT plain upload: the one-shot path warns it "might
  # take some time and then fail" on 100+ GiB folders. This one is RESUMABLE
  # (tracks state in .cache/huggingface inside the folder) and parallel, so a
  # dropped connection costs minutes, not the whole transfer.
  # 3 workers, not 8: with 8 the transfer stalled dead (byte-identical progress
  # for minutes, ~0.5 MB/s on the wire) on 2026-08-16. Fewer parallel LFS
  # streams held ~15 MB/s steadily.
  hf upload-large-folder "$ORG/$a" "$E/$a" --repo-type model --num-workers 3 --exclude ".cache/*" --exclude "*.DS_Store"

  # verify the Hub copy against local checksums before anyone downloads it
  echo "  --- verifying uploaded checksums ---"
  hf cache verify "$ORG/$a" --local-dir "$E/$a" --fail-on-missing-files \
    && echo "  ✅ checksums match" || { echo "  !! VERIFY FAILED — do not publish"; exit 1; }
  echo "  live: https://huggingface.co/$ORG/$a"
done
