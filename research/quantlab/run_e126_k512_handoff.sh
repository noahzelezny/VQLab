#!/bin/sh
# E126 — take the M4's d2/K512 rung and run it through the M3 instrument.
# Waits for the artifact to APPEAR AND STABILISE (index present, byte total
# unchanged across two 60s samples) rather than racing the peer's build.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
L=logs_live_e126_k512.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

ART=""
while [ -z "$ART" ]; do
  for c in "$E"/e119-27b-dense-d2k512* "$E"/e126-27b-dense-d2K512* "$E"/*d2K512*-vq "$E"/*d2k512*-vq; do
    [ -f "$c/model.safetensors.index.json" ] || continue
    case "$(basename $c)" in *-packed) continue;; esac
    ART="$c"; break
  done
  [ -z "$ART" ] && sleep 60
done
say "candidate: $(basename "$ART") — waiting for size to stabilise"
a=0; b=1
while [ "$a" != "$b" ]; do
  a=$(du -sk "$ART" | cut -f1); sleep 60; b=$(du -sk "$ART" | cut -f1)
done
say "stable at $((a/1024)) MiB"

say "zero-scan"
$V - <<PY 2>&1 | tee -a $L
import mlx.core as mx, json, pathlib
art=pathlib.Path("$ART")
amap=json.load(open(art/"model.safetensors.index.json"))["weight_map"]
by={}
for k,v in amap.items(): by.setdefault(v,[]).append(k)
dead=[]
for sh in sorted(by):
    with mx.stream(mx.cpu):
        T=mx.load(str(art/sh)); mx.eval(list(T.values()))
    dead += [k for k in by[sh] if float(mx.max(mx.abs(T[k].astype(mx.float32))).item())==0.0]
    del T; mx.clear_cache()
print(f"zero-scan: {len(dead)} all-zero tensors")
raise SystemExit(1 if dead else 0)
PY
[ $? -eq 0 ] || { say "ZERO-SCAN FAILED — stopping"; exit 1; }

say "gate"
$V verify_artifact.py --artifact "$ART" --src "$E/Qwen--Qwen3.8-27B" \
   --family qwen3_8_dense --outlier 3.0 2>&1 | tee -a $L | tail -5

PK="${ART}-packed"
say "pack (9-bit codes, genuinely packable unlike K256's 8-bit)"
$V pack_dense.py --src "$ART" --out "$PK" 2>&1 | tee -a $L | tail -2
SCORED="$PK"; [ -f "$PK/model.safetensors.index.json" ] || SCORED="$ART"
say "scoring $SCORED"

$V preflight_ram.py "$SCORED" 2>&1 | tee -a $L | grep -q "\-> OK" && $V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m,t = load("$SCORED")
print("SMOKE OK:", repr(generate(m,t,prompt="The capital of France is", max_tokens=8)))
PY

$V - <<PY 2>&1 | tee -a $L
import pathlib, mlx.core as mx, math
from mlx_lm.utils import load
p=pathlib.Path("$SCORED")
tot=sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"MEASURED SIZE {p.name}: {tot} bytes = {tot/2**30:.3f} GiB")
m,t=load("$SCORED")
ids=t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos=getattr(t,"bos_token_id",None)
if bos is not None and (not ids or ids[0]!=bos): ids=[bos]+ids[:2048]
lg=m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt=mx.array(ids[1:]); lse=mx.logsumexp(lg,axis=-1)
pk=mx.take_along_axis(lg,tgt[:,None].astype(mx.int64),axis=-1)[:,0]
print("PPL", math.exp(float(mx.mean(lse-pk).item())))
PY
$V kl_damage.py score --model "$SCORED" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -3
say "E126 DONE — bar: KL<=36.7 AND ppl<5.2055 by >=0.02 at <=14.80 GiB (29ddfb2). refs q4 45.842/5.2055/14.094; E124 40.327/5.2330/13.596. 6c branches: consistent=one-off, inverts-near-q4=near-lossless, far-from-q4=INCONCLUSIVE"
