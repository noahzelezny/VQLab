#!/bin/sh
# E130 arm 2 (d4/K4096, 3.00 bpw, 11.60 GiB projected) — the rate twin of
# arm 1 (d2/K64). The chain was stopped after arm 1's pack, so this replays
# the loop body from run_e130_rate_twin.sh for d=4/K=4096 ONLY, verbatim in
# step order and instrument, so the two arms stay comparable.
# Registered branches (d05d111, unchanged): d2 wins / d4 wins / INSIDE THE
# FLOOR -> indistinguishable. Floor 0.0447 ppl / 2.085 mnats is INHERITED
# from d2/K256 and is flagged as such per III.12 (121f639).
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
BASE="$E/qwen38-27b-rungs/q4"
L=logs_live_e130_twin.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }
d=4; K=4096
FIT="$E/e130-27b-d${d}K${K}"; ART="$FIT-vq"
say "=== arm d${d}/K${K}  (3.00 bpw, 11.60 GiB projected)"
[ -f "$FIT/config.json" ] || $V fit_dense_vq.py --family qwen3_8 --src "$SRC" \
   --out "$FIT" --k "$K" --dim "$d" --relerr-abort 0.90 >> $L 2>&1 \
   || { say "FIT FAILED d${d}/K${K}"; exit 1; }
grep "fit 192 tensors" $L | tail -1 | tee -a $L
say "collapses so far: $(grep -c 'relerr 1.0000' $L)"
[ -f "$ART/model.safetensors.index.json" ] || $V build_dense_vq.py --family qwen3_8 \
   --base "$BASE" --mlp "$FIT" --out "$ART" >> $L 2>&1 || { say "BUILD FAILED"; exit 1; }
$V verify_artifact.py --artifact "$ART" --src "$SRC" --family qwen3_8_dense \
   --outlier 3.0 2>&1 | tee -a $L | tail -3
PK="$ART-packed"
$V pack_dense.py --src "$ART" --out "$PK" 2>&1 | tee -a $L | tail -1
S="$PK"; [ -f "$PK/model.safetensors.index.json" ] || S="$ART"
$V preflight_ram.py "$S" 2>&1 | tee -a $L | grep -q "\-> OK" && $V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m,t = load("$S")
print("SMOKE OK:", repr(generate(m,t,prompt="The capital of France is", max_tokens=8)))
PY
$V - <<PY 2>&1 | tee -a $L
import pathlib, mlx.core as mx, math
from mlx_lm.utils import load
p=pathlib.Path("$S")
print(f"MEASURED SIZE {p.name}: {sum(f.stat().st_size for f in p.glob('*.safetensors'))/2**30:.3f} GiB")
m,t=load("$S")
ids=t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos=getattr(t,"bos_token_id",None)
if bos is not None and (not ids or ids[0]!=bos): ids=[bos]+ids[:2048]
lg=m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt=mx.array(ids[1:]); lse=mx.logsumexp(lg,axis=-1)
pk=mx.take_along_axis(lg,tgt[:,None].astype(mx.int64),axis=-1)[:,0]
print("PPL", math.exp(float(mx.mean(lse-pk).item())))
PY
$V kl_damage.py score --model "$S" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -3
say "=== arm d${d}/K${K} done — E130 COMPLETE"
