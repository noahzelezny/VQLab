#!/bin/sh
# E128 run C — 27B R3 candidate AND the decisive slope test.
# 12-bit codes at d2 = 6.00 bpw, projected 17.58 GiB (-33% vs the q8 we just
# measured at 26.341 GiB / 1.641 mnats). The d2 slope fitted to K256 and K512
# predicts ~18 mnats here — 11x the q8 bar. If it lands near 18, R3 is out of
# reach at any size that saves real bytes and we say so. If it lands far
# below, the curve steepens and R3 is live. Extrapolated four bits beyond two
# points, so the prediction is the thing being tested, not an assumption.
# 12 bits is NOT byte-aligned, so pack_dense does real work here.
# Small-case verified before writing this: L30 d2/K512 -> relerr 0.0592 in 49s.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
BASE="$E/qwen38-27b-rungs/q4"
FIT="$E/e124-27b-dense-d2K4096"
ART="$E/e124-27b-dense-d2K4096-vq"
PK="$E/e124-27b-dense-d2K4096-vq-packed"
L=logs_live_e124_d2.log

echo "########## E124 FIT (d2 K4096, layers 0-63) — 27B R3 candidate, decisive slope test $(date '+%H:%M:%S')" | tee -a $L
$V fit_dense_vq.py --family qwen3_8 --src "$SRC" --out "$FIT" \
   --k 4096 --dim 2 --relerr-abort 0.90 >> $L 2>&1 \
   || { echo "FIT FAILED — see $L" | tee -a $L; exit 1; }
tail -2 $L
grep -c "relerr 1.0000" $L | sed 's/^/collapses in fit log: /' | tee -a $L

echo "########## BUILD" | tee -a $L
$V build_dense_vq.py --family qwen3_8 --base "$BASE" --mlp "$FIT" --out "$ART" 2>&1 | tee -a $L | tail -3
[ -f "$ART/model.safetensors.index.json" ] || { echo "ABORT: no artifact" | tee -a $L; exit 1; }

echo "########## GATE" | tee -a $L
$V verify_artifact.py --artifact "$ART" --src "$SRC" --family qwen3_8_dense \
   --outlier 3.0 2>&1 | tee -a $L | tail -6

# PACK banner derived from K, not hardcoded. pack_dense is a NO-OP only when
# the index width is a whole number of bytes (bits % 8 == 0, i.e. K=256, 65536).
# At K=4096 the index is 12 bits and pack_dense does real work; a banner that
# says otherwise invites quoting the UNPACKED size, which III.8 forbids.
PACK_BITS=$($V -c "import sys;print(max(1,(int(sys.argv[1])-1).bit_length()))" "4096")
if [ $((PACK_BITS % 8)) -eq 0 ]; then
  echo "########## PACK (pack_dense; ${PACK_BITS} bits at K=4096 is BYTE-ALIGNED — pack_dense SKIPS it by design (cb4fb9b); the unpacked artifact IS the final size)" | tee -a $L
else
  echo "########## PACK (pack_dense; ${PACK_BITS} bits at K=4096 is NOT byte-aligned — pack_dense DOES real work. Quote the PACKED size only (III.8))" | tee -a $L
fi
$V pack_dense.py --src "$ART" --out "$PK" 2>&1 | tee -a $L | tail -3

for A in "$ART" "$PK"; do
  echo "--- preflight + III.11 smoke: $(basename $A)" | tee -a $L
  $V preflight_ram.py "$A" 2>&1 | tee -a $L | grep -q "\-> OK" || { echo "SKIP smoke (too big)" | tee -a $L; continue; }
  $V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m, t = load("$A")
print("SMOKE OK:", repr(generate(m, t, prompt="The capital of France is", max_tokens=8)))
PY
  $V - <<PY 2>&1 | tee -a $L
import pathlib
p = pathlib.Path("$A")
tot = sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"MEASURED SIZE {p.name}: {tot} bytes = {tot/2**30:.3f} GiB")
PY
done

# K256 d2 is byte-aligned so pack_dense produces no saving and may emit
# nothing; score whichever artifact actually exists, and SAY which.
SCORED="$PK"; [ -f "$PK/model.safetensors.index.json" ] || SCORED="$ART"
echo "########## SCORE (scoring $SCORED — E95 instrument)" | tee -a $L
$V - <<PY 2>&1 | tee -a $L
import mlx.core as mx, math
from mlx_lm.utils import load
m, t = load("$SCORED")
ids = t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos = getattr(t, "bos_token_id", None)
if bos is not None and (not ids or ids[0] != bos): ids = [bos] + ids[:2048]
lg = m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt = mx.array(ids[1:]); lse = mx.logsumexp(lg, axis=-1)
pk = mx.take_along_axis(lg, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
print("PPL", math.exp(float(mx.mean(lse - pk).item())))
PY
$V kl_damage.py score --model "$SCORED" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -4
echo "########## E124 DONE $(date '+%H:%M:%S') — PRIZE if KL<45.842 at <=14.09 GiB; STRONG if 45-90; NULL if >=187.765. refs: q2 1426.9@7.9G, q3 187.8@11.0G, q4 45.8@14.09G, bf16_ppl 5.2249" | tee -a $L
