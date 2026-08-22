#!/bin/sh
# E115 — can we reproduce the shipped 2.4 (2.7655) ON PURPOSE?
#
# THE QUESTION. Every shipped artifact was fit 08-16 with RANDOM seeding;
# every refit was fit 08-21 with kmeans++ (the default flipped 08-18). Those
# are init A/Bs, and they split by K:
#     K256  (111.6G): random 2.7655  vs  ++ 2.8057   -> ++ HURTS
#     K2048 (143.7G): random 2.3519  vs  ++ 2.3410   -> ++ HELPS
# E107-E110 explain it per-tensor: ++ buys the bulk by selling the tail, and
# only where centroids are scarce. That has NEVER been tested end-to-end.
# This is that test: E92 with one flag changed.
#
# GEOMETRY is reconstructed from rotlab--397B-flatk256-refit/config.json
# (d4, k256, group 64, 171 modules) because the E92 fit ran on M4 and left no
# script on this box. --iters and --expert-chunk are NOT recorded in config;
# both are left at their defaults, which is what E92 would have used. They
# change memory and convergence budget, not the objective.
#
# BASE is struct6-tail3x3, the SAME base E89 used and the only artifact with
# 171 2-bit switch_mlp modules -- exactly the 171 the refit VQ'd. Do NOT use
# the shipped 2.4 as base: its experts are ALREADY codes, so the fitter finds
# 0 expert tensors and exits. (The refit's own config LOOKS like the shipped
# 2.4's because VQ'd modules are dropped from the quantization block -- an
# output stops resembling its own base. That misread cost one launch.)
#
# --relerr-abort 0.90, NOT the 0.35 default: legitimate d4K256 tensors on this
# model reach 0.4332 (L01 up_proj, shipped 2.4 gate). 0.35 would abort a
# healthy fit at L00. 0.90 is the collapse bar.
#
# PRE-REGISTERED, written before any number exists:
#   CONFIRMED  wikitext <= 2.7655 -> the recipe is recovered and the mechanism
#              is validated end-to-end, not just per-tensor.
#   FALSIFIED  wikitext ~ 2.8057  -> init is NOT what separates shipped 2.4
#              from the refit, and something else in the 08-16 -> 08-21 drift
#              is. That branch is the more valuable one: it means no refit is
#              trustworthy until we find it.
#   Anything between is REPORTED AS BETWEEN. No rounding toward a hypothesis.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
BASE="$E/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3"
SRC="$E/Qwen--Qwen3.5-397B-A17B-bf16"
SHIPPED="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
OUT="rotlab--397B-flatk256-randinit"
L=logs_live_e115_randinit.log

echo "########## E115 FIT $OUT (d4 k256, --init random)" | tee -a $L
date '+start %H:%M:%S' | tee -a $L
$V vq_397b_codes.py --family qwen3_5 --base "$BASE" --src "$SRC" \
   --out "$E/$OUT" --vq-layers 0-56 --k 256 --dim 4 --group 64 \
   --init random --relerr-abort 0.90 >> $L 2>&1 \
   || { echo "FIT FAILED $OUT -- see $L" | tee -a $L; exit 1; }
tail -3 $L
# HARD GATE: a failed fit must never fall through to pack/score. The first
# launch did exactly that -- `python | tee | tail` reports TAIL's exit status,
# which is 0 even when python died, so the whole chain ran on empty dirs.
[ -f "$E/$OUT/model.safetensors.index.json" ] || {
  echo "ABORT: fit produced no index -- refusing to pack/score nothing" | tee -a $L; exit 1; }

$V add_model_file.py --artifact "$E/$OUT" >/dev/null 2>&1
cp "$SHIPPED/tokenizer.json" "$SHIPPED/tokenizer_config.json" "$E/$OUT/"
echo "########## GATE (unpacked)" | tee -a $L
$V verify_artifact.py --artifact "$E/$OUT" --src "$SRC" --family qwen3_5 \
   --outlier 3.0 2>&1 | tee -a $L | tail -5

echo "########## PACK" | tee -a $L
$V pack_artifact.py --src "$E/$OUT" --out "$E/$OUT-packed" 2>&1 | tee -a $L | tail -1
cp "$SHIPPED/tokenizer.json" "$SHIPPED/tokenizer_config.json" "$E/$OUT-packed/"
$V graft_vision.py --artifact "$E/$OUT-packed" --src "$SRC" \
   --prefixes model.visual \
   --copy-config-keys vision_config,image_token_id 2>&1 | tee -a $L | tail -2
$V check_vision.py  --artifact "$E/$OUT-packed" --src "$SRC" 2>&1 | tee -a $L | tail -1
$V check_release.py --artifact "$E/$OUT-packed" 2>&1 | tee -a $L | tail -1
$V - <<PY 2>&1 | tee -a $L
import pathlib
p = pathlib.Path("$E/$OUT-packed")
t = sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"SIZE post-graft whole-artifact: {t} bytes = {t/2**30:.3f} GiB")
PY

echo "########## SCORE" | tee -a $L
$V referee/score_streaming.py --model "$E/$OUT-packed" --corpus referee/referee_corpus.txt 2>&1 | tee -a $L | tail -1
$V referee/score_streaming.py --model "$E/$OUT-packed" --corpus referee/referee_corpus_code.txt 2>&1 | tee -a $L | tail -1
date '+end %H:%M:%S' | tee -a $L
echo "########## E115 DONE — CONFIRMED <=2.7655 / FALSIFIED ~2.8057; refs: shipped 2.4 2.7655/2.6383, flatk256-refit(++) 2.8057/2.6447, both @111.617 GiB" | tee -a $L
