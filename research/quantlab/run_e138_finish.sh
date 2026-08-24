#!/bin/sh
# E138 finish chain: fit is done (b7e2686 pre-registration). build -> outlier
# gate -> pack -> III.11 smoke -> ppl -> KL. Same instrument as E124/E126/E128C.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
BASE="$E/qwen38-27b-rungs/q4"
FIT="$E/e138-27b-dense-d4K65536"
ART="$FIT-vq"
L=logs_live_e138_finish.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

say "=== BUILD"
[ -f "$ART/model.safetensors.index.json" ] || $V build_dense_vq.py --family qwen3_8 \
   --base "$BASE" --mlp "$FIT" --out "$ART" >> $L 2>&1 || { say "BUILD FAILED"; exit 1; }

say "=== VERIFY / OUTLIER GATE"
$V verify_artifact.py --artifact "$ART" --src "$SRC" --family qwen3_8_dense \
   --outlier 3.0 2>&1 | tee -a $L | tail -4

say "=== PACK"
PK="$ART-packed"
$V pack_dense.py --src "$ART" --out "$PK" 2>&1 | tee -a $L | tail -2
S="$PK"; [ -f "$PK/model.safetensors.index.json" ] || S="$ART"

say "=== PREFLIGHT + SMOKE (III.11)"
$V preflight_ram.py "$S" 2>&1 | tee -a $L | tail -1
$V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m,t = load("$S")
print("SMOKE OK:", repr(generate(m,t,prompt="The capital of France is", max_tokens=8)))
PY

say "=== SIZE + PPL"
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

say "=== KL"
$V kl_damage.py score --model "$S" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -3
say "=== E138 CHAIN COMPLETE. bar: E124 d2/K256 40.327 mnats / 90.10% / ppl 5.2330 @ 13.596 GiB (size model predicts 13.594 packed)"
