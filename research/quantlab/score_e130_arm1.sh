#!/bin/sh
# Score E130 arm 1 (d2/K64, 3.00 bpw). The rate-twin chain was stopped after
# arm 1's pack, so this replays EXACTLY the score block from
# run_e130_rate_twin.sh — same instrument, so arm 1 and arm 2 stay comparable.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
S="$E/e130-27b-d2K64-vq-packed"
L=logs_live_e130_twin.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }
say "=== SCORING arm d2/K64 (resumed; chain stopped after pack)"
$V preflight_ram.py "$S" 2>&1 | tee -a $L | grep -q "\-> OK" || { say "PREFLIGHT FAILED — abort"; exit 1; }
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
say "=== arm d2/K64 SCORED"
