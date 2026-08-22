#!/bin/sh
# E128 A+B — the missing 8-bit comparators. No R3 rung is interpretable
# without them, so they run before any R3 fit.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
L=logs_live_e128_comparators.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

say "=== A: 35B affine 8-bit on kl_cache_qwen36"
$V kl_damage.py score --model "$E/mlx-community--Qwen3.6-35B-A3B-8bit" \
   --cache-dir "$E/kl_cache_qwen36" 2>&1 | tee -a $L | tail -3
$V - <<PY 2>&1 | tee -a $L
import pathlib
p=pathlib.Path("$E/mlx-community--Qwen3.6-35B-A3B-8bit")
print(f"MEASURED SIZE {p.name}: {sum(f.stat().st_size for f in p.glob('*.safetensors'))/2**30:.3f} GiB")
PY

say "=== B: build a 27B q8, then score it"
Q8="$E/qwen38-27b-rungs/q8"
if [ ! -f "$Q8/model.safetensors.index.json" ]; then
  $V convert_qwen38_mixed.py --src "$E/Qwen--Qwen3.8-27B" \
     --out-root "$E/qwen38-27b-rungs" --name q8 \
     --mlp-bits 8 --mlp-group-size 64 --linattn-bits 8 \
     --fullattn-bits 8 --embed-bits 8 >> $L 2>&1 \
     || { say "q8 BUILD FAILED — see $L"; exit 1; }
fi
$V - <<PY 2>&1 | tee -a $L
import pathlib
p=pathlib.Path("$Q8")
print(f"MEASURED SIZE q8: {sum(f.stat().st_size for f in p.glob('*.safetensors'))/2**30:.3f} GiB")
PY
$V kl_damage.py score --model "$Q8" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -3
$V - <<PY 2>&1 | tee -a $L
import mlx.core as mx, math
from mlx_lm.utils import load
m,t=load("$Q8")
ids=t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos=getattr(t,"bos_token_id",None)
if bos is not None and (not ids or ids[0]!=bos): ids=[bos]+ids[:2048]
lg=m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt=mx.array(ids[1:]); lse=mx.logsumexp(lg,axis=-1)
pk=mx.take_along_axis(lg,tgt[:,None].astype(mx.int64),axis=-1)[:,0]
print("PPL", math.exp(float(mx.mean(lse-pk).item())))
PY
say "E128 A+B DONE — these are the R3 targets. refs: 27B q4 45.842/5.2055@14.094; 35B q4 78.557@19.0"
