#!/bin/sh
# E92 + E93 gate chains. The FITS and the PACKING were done on M4; this box
# only gates and scores them (E47: never gate your own artifact).
#
# Differs from run_e89_m3.sh: no fit, no pack step. Starts at verify on the
# unpacked dir, then grafts/checks/scores the already-packed twin.
#
# graft_vision now defaults --copy-config-keys ON (d143d8b), so these two get
# vision_config automatically — every earlier chain omitted the flag and left
# the artifacts needing a hand-graft. Left explicit here anyway so the chain
# reads as self-documenting rather than relying on a default.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.5-397B-A17B-bf16"
SHIPPED="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
L=logs_live_e92e93.log

$V check_scripts_sync.sh >/dev/null 2>&1 || sh check_scripts_sync.sh 2>&1 | tee -a $L | tail -2

gate() {
  raw="$1"; pk="$2"
  echo "########## GATE $pk" | tee -a $L
  $V verify_artifact.py --artifact "$E/$raw" --src "$SRC" --family qwen3_5 \
     --outlier 3.0 2>&1 | tee -a $L | tail -4
  cp "$SHIPPED/tokenizer.json" "$SHIPPED/tokenizer_config.json" "$E/$pk/" 2>/dev/null
  $V graft_vision.py --artifact "$E/$pk" --src "$SRC" \
     --prefixes model.visual \
     --copy-config-keys vision_config,image_token_id 2>&1 | tee -a $L | tail -2
  $V check_vision.py --artifact "$E/$pk" --src "$SRC" 2>&1 | tee -a $L | tail -1
  $V check_release.py --artifact "$E/$pk" 2>&1 | tee -a $L | tail -1
  # whole-artifact bytes, stamped POST-graft, so this size is never confused
  # with a pack-log subset figure or a pre-graft number
  $V - <<PY 2>&1 | tee -a $L
import pathlib
p = pathlib.Path("$E/$pk")
t = sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"SIZE post-graft whole-artifact: {t} bytes = {t/2**30:.3f} GiB")
PY
  $V referee/score_streaming.py --model "$E/$pk" --corpus referee/referee_corpus.txt 2>&1 | tee -a $L | tail -1
  $V referee/score_streaming.py --model "$E/$pk" --corpus referee/referee_corpus_code.txt 2>&1 | tee -a $L | tail -1
}

gate "rotlab--397B-flatk256-refit" "rotlab--397B-flatk256-refit-packed"
gate "rotlab--397B-flatk512"       "rotlab--397B-flatk512-packed"

echo "########## E92/E93 DONE — refs: shipped 2.2 3.1706 @100.9G, d8K16384 3.0591/2.6728 @100.97G, cheapshallow-2.3 2.779/2.6479 @107.9G" | tee -a $L
