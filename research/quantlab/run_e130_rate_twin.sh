#!/bin/sh
# E130 — the d-vs-K rate twin at 3.00 bpw. Both arms are 11.60 GiB by
# construction: 6 bits per 2 weights and 12 bits per 4 weights are the SAME
# RATE. This is the experiment that licenses (or kills) any claim about which
# of d or K is the better place to spend a fixed budget on a dense model.
# The exact twin of R1 would be d2/K256 vs d4/K65536, but K65536 is ~40-60h
# and gives only 340 subvectors per centroid; this band is ~4h and is not
# sample-starved. Band caveat is explicit: it answers d-vs-K at 3.00 bpw, NOT
# at R1's 4.00 bpw.
#
# PRE-REGISTERED, before either arm runs:
#   d2 WINS  -> spending on a finer subvector beats a bigger codebook at fixed
#               rate on dense. Supports the R1/R2 geometry choice by analogy,
#               one band down; does NOT prove it at 4.00 bpw.
#   d4 WINS  -> we have been leaving quality on the table at R1/R2, and the
#               expensive K65536 twin becomes worth its 40-60h.
#   INSIDE THE FLOOR -> report as indistinguishable. The 27B ppl floor is
#               0.0447 and the KL floor 2.085 mnats (6f, n=3). A difference
#               smaller than that is NOT a result in either direction.
# Waits for run C rather than contending — two fits on one box invalidated a
# timing measurement earlier today.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
BASE="$E/qwen38-27b-rungs/q4"
L=logs_live_e130_twin.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

while pgrep -f "fit_dense_vq.py .*d2K4096" >/dev/null; do sleep 120; done
say "run C clear — starting the rate twin"

for arm in d2:64 d4:4096; do
  d=${arm%%:*}; d=${d#d}; K=${arm##*:}
  FIT="$E/e130-27b-d${d}K${K}"; ART="$FIT-vq"
  say "=== arm d${d}/K${K}  (3.00 bpw, 11.60 GiB projected)"
  [ -f "$FIT/config.json" ] || $V fit_dense_vq.py --family qwen3_8 --src "$SRC" \
     --out "$FIT" --k "$K" --dim "$d" --relerr-abort 0.90 >> $L 2>&1 \
     || { say "FIT FAILED d${d}/K${K}"; continue; }
  grep "fit 192 tensors" $L | tail -1 | tee -a $L
  say "collapses so far: $(grep -c 'relerr 1.0000' $L)"
  [ -f "$ART/model.safetensors.index.json" ] || $V build_dense_vq.py --family qwen3_8 \
     --base "$BASE" --mlp "$FIT" --out "$ART" >> $L 2>&1 || { say "BUILD FAILED"; continue; }
  $V verify_artifact.py --artifact "$ART" --src "$SRC" --family qwen3_8_dense \
     --outlier 3.0 2>&1 | tee -a $L | tail -3
  PK="$ART-packed"
  $V pack_dense.py --src "$ART" --out "$PK" 2>&1 | tee -a $L | tail -1
  S="$PK"; [ -f "$PK/model.safetensors.index.json" ] || S="$ART"
  $V preflight_ram.py "$S" 2>&1 | grep -q "\-> OK" && $V - <<PY 2>&1 | tee -a $L
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
  say "=== arm d${d}/K${K} done"
done
say "E130 DONE — both arms are 3.00 bpw / 11.60 GiB. Floor: KL 2.085 mnats, ppl 0.0447 (6f). A gap inside that is NOT a result. refs: q3 187.765@10.963, q4 45.842@14.094"
