#!/bin/sh
# E124 — 27B dense at 4-bit SIZE (d2), chasing q4's quality. Noah's stated target.
# Small-case verified before writing this: L30 d2/K512 -> relerr 0.0592 in 49s.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
BASE="$E/qwen38-27b-rungs/q4"
FIT="$E/e124-27b-dense-d2K256"
ART="$E/e124-27b-dense-d2K256-vq"
PK="$E/e124-27b-dense-d2K256-vq-packed"
L=logs_live_e124_d2.log

echo "########## E124 FIT (d2 K256, layers 0-63) $(date '+%H:%M:%S')" | tee -a $L
$V fit_dense_vq.py --family qwen3_8 --src "$SRC" --out "$FIT" \
   --k 256 --dim 2 --relerr-abort 0.90 >> $L 2>&1 \
   || { echo "FIT FAILED — see $L" | tee -a $L; exit 1; }
tail -2 $L
grep -c "relerr 1.0000" $L | sed 's/^/collapses in fit log: /' | tee -a $L

echo "########## BUILD" | tee -a $L
$V build_dense_vq.py --family qwen3_8 --base "$BASE" --mlp "$FIT" --out "$ART" 2>&1 | tee -a $L | tail -3
[ -f "$ART/model.safetensors.index.json" ] || { echo "ABORT: no artifact" | tee -a $L; exit 1; }

echo "########## GATE" | tee -a $L
$V verify_artifact.py --artifact "$ART" --src "$SRC" --family qwen3_8_dense \
   --outlier 3.0 2>&1 | tee -a $L | tail -6

echo "########## PACK (pack_dense; 8 bits at K256 is BYTE-ALIGNED — pack_dense SKIPS it by design (cb4fb9b); the unpacked artifact IS the final size)" | tee -a $L
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
