#!/bin/sh
# E112 gate chain: body-only tail-weighted K256 (rotlab--397B-flatk256-bodytailw4).
#
# READ BEFORE INTERPRETING THE GATE OUTPUT:
# mean relerr is EXPECTED to be worse (fit reported 0.3469 vs the K256 refit's
# ~0.3116). That +0.035 is THE TRADE BEING BOUGHT — body layers give up bulk
# accuracy to buy the tail (E102/E109). Reading a raised mean as a damaged fit
# repeats E101 with the sign flipped. The check that still works is the OUTLIER
# gate (median x3), which is relative and immune to a uniform shift.
#
# III.10 (generate one token through the fused path) is NOT in this chain:
# the packed artifact is ~111 GiB against M3's 96 GiB (rule III.4). It runs on
# M4 before any claim is published.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.5-397B-A17B-bf16"
SHIPPED="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
RAW="rotlab--397B-flatk256-bodytailw4"
PK="$RAW-packed"
L=logs_live_e112_chain.log

sh check_scripts_sync.sh 2>&1 | tail -1 | tee -a $L

echo "########## VERIFY $RAW" | tee -a $L
$V verify_artifact.py --artifact "$E/$RAW" --src "$SRC" --family qwen3_5 \
   --outlier 3.0 2>&1 | tee -a $L | tail -6

$V add_model_file.py --artifact "$E/$RAW" >/dev/null 2>&1
echo "########## PACK" | tee -a $L
$V pack_artifact.py --src "$E/$RAW" --out "$E/$PK" 2>&1 | tee -a $L | tail -2
cp "$SHIPPED/tokenizer.json" "$SHIPPED/tokenizer_config.json" "$E/$PK/" 2>/dev/null

echo "########## GRAFT + CHECKS" | tee -a $L
$V graft_vision.py --artifact "$E/$PK" --src "$SRC" --prefixes model.visual \
   --copy-config-keys vision_config,image_token_id 2>&1 | tee -a $L | tail -2
$V check_vision.py --artifact "$E/$PK" --src "$SRC" 2>&1 | tee -a $L | tail -1
$V check_release.py --artifact "$E/$PK" 2>&1 | tee -a $L | tail -1
$V fix_index_total_size.py --artifact "$E/$PK" 2>&1 | tee -a $L | tail -2
$V - <<PY 2>&1 | tee -a $L
import pathlib
p = pathlib.Path("$E/$PK")
t = sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"SIZE post-graft whole-artifact: {t} bytes = {t/2**30:.3f} GiB")
PY

echo "########## SCORE (streaming; works above RAM)" | tee -a $L
$V referee/score_streaming.py --model "$E/$PK" --corpus referee/referee_corpus.txt 2>&1 | tee -a $L | tail -1
$V referee/score_streaming.py --model "$E/$PK" --corpus referee/referee_corpus_code.txt 2>&1 | tee -a $L | tail -1

echo "########## E112 DONE — refs: flatk256-refit 2.8057/2.6447 @111.617G (primary, isolates the weighting); shipped 2.4 2.7655/2.6383 @111.617G (ship bar)" | tee -a $L
