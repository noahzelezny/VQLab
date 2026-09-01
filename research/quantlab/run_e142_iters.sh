#!/bin/sh
# E142 — iters lever at d2/K512. Two arms on the M3, seed 1234, differing ONLY
# in --iters. Full chain per arm: fit -> build -> gate -> pack -> smoke -> ppl -> KL.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
BASE="$E/qwen38-27b-rungs/q4"
L=logs_live_e142_iters.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

for it in 10 30; do
  FIT="$E/e142-27b-d2K512-iters$it"; ART="$FIT-vq"
  say "===== ARM iters=$it"
  [ -f "$FIT/config.json" ] || $V fit_dense_vq.py --family qwen3_8 --src "$SRC" \
     --out "$FIT" --k 512 --dim 2 --iters "$it" --seed 1234 --relerr-abort 0.90 >> $L 2>&1 \
     || { say "FIT FAILED iters=$it"; continue; }
  grep "fit 192 tensors" $L | tail -1 | tee -a $L
  [ -f "$ART/model.safetensors.index.json" ] || $V build_dense_vq.py --family qwen3_8 \
     --base "$BASE" --mlp "$FIT" --out "$ART" >> $L 2>&1 || { say "BUILD FAILED"; continue; }
  $V verify_artifact.py --artifact "$ART" --src "$SRC" --family qwen3_8_dense \
     --outlier 3.0 2>&1 | tee -a $L | tail -3
  PK="$ART-packed"
  $V pack_dense.py --src "$ART" --out "$PK" 2>&1 | tee -a $L | tail -2
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
  say "===== ARM iters=$it DONE"
done
say "E142 COMPLETE. bar E126: 14.592 GiB / KL 33.095 / top-1 91.10% / ppl 5.194289 (M4, unseeded, iters10)"
