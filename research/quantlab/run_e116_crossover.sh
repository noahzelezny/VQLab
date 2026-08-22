#!/bin/sh
# E118 (script filename says e116 — E116 was already the peer's SMB
# round-trip; collision found after the E117 chain was launched and the
# watcher armed on this filename. NEVER rename a file an armed watcher will
# exec.) — WHERE IS THE INIT CROSSOVER IN K?
#
# PRE-REGISTERED before any number exists. Known signs, end-to-end, 397B:
#     K256  (111.6G): random 2.7655 beats ++ 2.8057   (E92 vs shipped 2.4)
#     K2048 (143.7G): ++ 2.3410 beats random 2.3519   (flagship refit)
# Mechanism (E107-E110): ++ buys the bulk by selling the tail, and only hurts
# where centroids are scarce. Somewhere in (256, 2048) the sign flips. K512
# bisects in log space, and we already hold a ++-seeded K512 artifact:
#     flatk512 (E93, ++): 2.5634 / 2.6123 @ 122.305 GiB
# So ONE new fit — K512 --init random, same base, same geometry — gives the
# pair. Same-size twin, one flag apart, both scored on the same referee.
#
# READINGS:
#   random < 2.5634 (by >= 0.005)  -> crossover is ABOVE 512; recipe so far:
#                                     random through K512, ++ by K2048.
#   random > 2.5634 (by >= 0.005)  -> crossover is BELOW 512; K256 is inside
#                                     the scarce band but K512 already is not.
#   |delta| < 0.005                -> AT the crossover / too close to call at
#                                     this instrument's resolution. Reported
#                                     as such, not rounded.
# CAVEAT, stated up front: E117 (banner says E115, same collision) (K256 random, running tonight) is the
# validity check for this whole framing. If E115 FALSIFIES init as the E92
# regression's cause, E118's result still stands as a measured pair but the
# "crossover" interpretation is void until the real cause is found.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
BASE="$E/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3"
SRC="$E/Qwen--Qwen3.5-397B-A17B-bf16"
SHIPPED="$E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
OUT="rotlab--397B-flatk512-randinit"
L=logs_live_e116_crossover.log

echo "########## E118 FIT $OUT (d4 k512, --init random)" | tee -a $L
date '+start %H:%M:%S' | tee -a $L
$V vq_397b_codes.py --family qwen3_5 --base "$BASE" --src "$SRC" \
   --out "$E/$OUT" --vq-layers 0-56 --k 512 --dim 4 --group 64 \
   --init random --relerr-abort 0.90 >> $L 2>&1 \
   || { echo "FIT FAILED $OUT -- see $L" | tee -a $L; exit 1; }
tail -3 $L
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
echo "########## E118 DONE — twin: flatk512(++) 2.5634/2.6123 @122.305 GiB. ABOVE-512 if random<2.5634 by 0.005; BELOW-512 if >; TOO-CLOSE if within." | tee -a $L
