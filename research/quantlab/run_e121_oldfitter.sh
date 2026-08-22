#!/bin/sh
# E121 — RUN THE 08-16 FITTER ITSELF.
#
# THE QUESTION Noah actually asked: how did the shipped 2.4 happen, when
# nothing we build today reaches it? E117 excluded seeding; E120 tests
# summation order per-tensor. This tests the whole thing end-to-end by
# running the ACTUAL CODE that produced the shipped artifact.
#
# fitter_0816_cdcdeab.py is `git show cdcdeab:vq_397b_codes.py` verbatim —
# the last commit touching the fitter before the 08-18/19 k-means changes.
# Its kmeans() is random init + one-hot matmul accumulation, fixed 2,000,000
# row chunks. Committed as a FILE (not just a git ref) so this run is
# reproducible after any rebase, and so the diff that matters is readable
# side-by-side with HEAD.
#
# NOTE: no --init flag (didn't exist), no --relerr-abort (didn't exist), no
# --tail-* (didn't exist). That is the point — this is the old tool, unedited.
# It also means NO abort guard: gate hard afterward, and read the fit log.
#
# PRE-REGISTERED:
#   REPRODUCES  wikitext <= 2.7700 -> the regression lives in the three
#               08-18/19 k-means commits. Bisect them next (E120 already
#               names summation order as the leading suspect).
#   DOES NOT    wikitext ~ 2.81+   -> the fitter FILE is not the cause. The
#               difference is then environmental (mlx version, base artifact,
#               or the source checkpoint), which is a bigger finding and
#               changes what "fitter vintage" has meant in every entry since
#               E94. Do not paper over this branch.
#   Between 2.7700 and 2.8057: partial — report as partial, bisect anyway.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
BASE="$E/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3"
SRC="$E/Qwen--Qwen3.5-397B-A17B-bf16"
SHIPPED="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
OUT="rotlab--397B-flatk256-fitter0816"
L=logs_live_e121_oldfitter.log

echo "########## E121 FIT $OUT (08-16 fitter cdcdeab, d4 k256)" | tee -a $L
date '+start %H:%M:%S' | tee -a $L
$V fitter_0816_cdcdeab.py --family qwen3_5 --base "$BASE" --src "$SRC" \
   --out "$E/$OUT" --vq-layers 0-56 --k 256 --dim 4 --group 64 >> $L 2>&1 \
   || { echo "FIT FAILED $OUT -- see $L" | tee -a $L; exit 1; }
tail -3 $L
[ -f "$E/$OUT/model.safetensors.index.json" ] || {
  echo "ABORT: fit produced no index" | tee -a $L; exit 1; }

$V add_model_file.py --artifact "$E/$OUT" >/dev/null 2>&1
cp "$SHIPPED/tokenizer.json" "$SHIPPED/tokenizer_config.json" "$E/$OUT/"
echo "########## GATE (unpacked) — the old fitter has NO abort; this is the only guard" | tee -a $L
$V verify_artifact.py --artifact "$E/$OUT" --src "$SRC" --family qwen3_5 \
   --outlier 3.0 2>&1 | tee -a $L | tail -6

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
echo "########## E121 DONE — REPRODUCES if <=2.7700; NOT-THE-FILE if ~2.81+. refs: shipped 2.4 2.7655/2.6383, ++ refit 2.8057/2.6447, random refit 2.8158/2.6347, all @111.617 GiB" | tee -a $L
