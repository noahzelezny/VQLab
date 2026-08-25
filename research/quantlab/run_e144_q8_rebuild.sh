#!/bin/sh
# E144 — rebuild the 27B q8 comparator with DEFAULT mlx_lm.convert settings.
# Spec: handoff/M3_AFTER_27B.md §6. Same scoring instrument as E124/E126/E128C/E138.
# PRE-REGISTERED, before any number exists: the rebuilt q8 should be SMALLER (~0.02 GiB)
# and WORSE (higher KL) than the incumbent 26.341 GiB / KL 1.641 mnats / top-1 98.08%,
# because it quantizes 23.6 M params the incumbent left at bf16.
# If it comes back BETTER or IDENTICAL: DO NOT ADOPT until the difference is explained.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
OUT="$E/qwen38-27b-rungs/q8-rebuilt"
OLD="$E/qwen38-27b-rungs/q8"
L=logs_live_e144.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

say "=== CONVERT (mlx_lm defaults; q_bits=8 is the ONLY deviation)"
rm -rf "$OUT"
$V -c "
from mlx_lm.convert import convert
convert(hf_path='$SRC', mlx_path='$OUT', quantize=True, q_bits=8)
print('convert done')" 2>&1 | tee -a $L | tail -5

say "=== ASSERT BEFORE SCORING (stop here if the conversion did not fix the defect)"
$V - "$OUT" "$OLD" <<'PY' 2>&1 | tee -a $L
import sys, glob
from safetensors import safe_open
def tensors(p):
    d={}
    for f in glob.glob(p+"/*.safetensors"):
        with safe_open(f,"numpy") as g:
            for k in g.keys(): d[k]=tuple(g.get_slice(k).get_shape())
    return d
new,old=tensors(sys.argv[1]),tensors(sys.argv[2])
print("tensor count  new %d  old %d   (q4/q6/VQ family = 1847)"%(len(new),len(old)))
ip=[k for k in new if "in_proj_a" in k]
sc=[k for k in ip if k.endswith("scales")]; bi=[k for k in ip if k.endswith("biases")]
print("in_proj_a: %d tensors, scales %d, biases %d"%(len(ip),len(sc),len(bi)))
fail=[]
if len(new)!=1847: fail.append("tensor count %d != 1847"%len(new))
if len(sc)!=48 or len(bi)!=48: fail.append("in_proj_a scales/biases %d/%d != 48/48"%(len(sc),len(bi)))
# Added on top of the handoff spec: non-attention tensors must be unchanged in shape,
# so "smaller and worse" is attributable to the 96 projections and not to some other
# default that also moved between the two conversions.
shared=[k for k in set(new)&set(old) if "linear_attn" not in k]
diff=[(k,old[k],new[k]) for k in shared if old[k]!=new[k]]
print("non-linear_attn tensors compared: %d   shape diffs: %d"%(len(shared),len(diff)))
for k,o,n in diff[:10]: print("   DIFF",k,o,"->",n)
if diff: fail.append("%d non-attention tensors changed shape"%len(diff))
if fail:
    print("ASSERT FAILED:")
    for f in fail: print("  -",f)
    sys.exit(1)
print("ASSERT PASS: 1847 tensors, in_proj_a quantized 48/48, non-attention geometry identical")
PY
[ $? -eq 0 ] || { say "ASSERT FAILED — stopping before scoring, per spec"; exit 1; }

say "=== OUTLIER GATE"
$V verify_artifact.py --artifact "$OUT" --src "$SRC" --family qwen3_8_dense \
   --outlier 3.0 2>&1 | tee -a $L | tail -4

say "=== SIZE + PPL  (same instrument as every other 27B row)"
$V - <<PY 2>&1 | tee -a $L
import pathlib, mlx.core as mx, math
from mlx_lm.utils import load
p=pathlib.Path("$OUT")
print(f"MEASURED SIZE {p.name}: {sum(f.stat().st_size for f in p.glob('*.safetensors'))/2**30:.3f} GiB")
m,t=load("$OUT")
ids=t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos=getattr(t,"bos_token_id",None)
if bos is not None and (not ids or ids[0]!=bos): ids=[bos]+ids[:2048]
lg=m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt=mx.array(ids[1:]); lse=mx.logsumexp(lg,axis=-1)
pk=mx.take_along_axis(lg,tgt[:,None].astype(mx.int64),axis=-1)[:,0]
print("PPL", math.exp(float(mx.mean(lse-pk).item())))
PY

say "=== KL"
$V kl_damage.py score --model "$OUT" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -3

say "=== E144 COMPLETE. incumbent: 26.341 GiB / KL 1.641 mnats / top-1 98.08%"
say "=== PRE-REG: expect SMALLER + WORSE. Better-or-identical => DO NOT ADOPT."
