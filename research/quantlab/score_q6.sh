#!/bin/sh
# E133 — score the 27B q6 affine comparator on the ORIGINAL venv, which is the
# instrument that scored E128C (d2/K4096) and every other 27B rung. q6 is
# affine and touches no VQ kernel, so venv choice is immaterial to the math;
# using the original preserves instrument continuity regardless (III.2).
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
S="$E/qwen38-27b-rungs/q6"
L=logs_live_q6.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }
say "=== E133 q6 (uniform 6-bit affine, group 64)"
$V preflight_ram.py "$S" 2>&1 | tee -a $L | tail -1
$V - <<PY 2>&1 | tee -a $L
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
say "=== q6 done. refs same instrument: q4 45.842@14.094 ppl 5.2055 | E128C d2/K4096 26.709@17.583 ppl 5.2417 | q8 1.641@26.341 | bf16 ppl 5.2249"
