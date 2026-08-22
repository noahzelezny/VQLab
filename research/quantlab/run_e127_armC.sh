#!/bin/sh
# E127 ARM C — the NOISE FLOOR. Identical settings to arm A (--iters 10),
# different init draw (MLX RNG is unseeded across processes, 6d). |A-C| is
# the seed-noise floor at this model and geometry; the A-vs-B effect is read
# against it. Waits for the A/B chain rather than running beside it.
# Original header follows.
# E127 — clean law-6 specimen on the dense 27B. Two arms, ONE knob (--iters),
# identical source and base bytes, built minutes apart so no provenance
# question can reopen the pair. Strictly sequential.
set -u
while pgrep -f run_e127_law6_specimen >/dev/null; do sleep 60; done
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
SRC="$E/Qwen--Qwen3.8-27B"
BASE="$E/qwen38-27b-rungs/q4"
L=logs_live_e127_law6.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

for arm in C:10; do
  n=${arm%%:*}; it=${arm##*:}
  FIT="$E/e127-27b-d2K256-armC-iters$it"; ART="$E/e127-27b-d2K256-armC-iters$it-vq"
  say "=== ARM C (--iters 10, noise-floor twin of arm A)"
  if [ ! -f "$FIT/config.json" ]; then
    $V fit_dense_vq.py --family qwen3_8 --src "$SRC" --out "$FIT" \
       --k 256 --dim 2 --iters "$it" --relerr-abort 0.90 >> $L 2>&1 \
       || { say "FIT FAILED arm $n"; exit 1; }
  fi
  grep "fit 192 tensors" $L | tail -1 | tee -a $L
  c=$(grep -c "relerr 1.0000" $L); say "arm $n collapses so far: $c"
  if [ ! -f "$ART/model.safetensors.index.json" ]; then
    $V build_dense_vq.py --family qwen3_8 --base "$BASE" --mlp "$FIT" \
       --out "$ART" >> $L 2>&1 || { say "BUILD FAILED arm $n"; exit 1; }
  fi
  say "--- gate arm $n"
  $V verify_artifact.py --artifact "$ART" --src "$SRC" \
     --family qwen3_8_dense --outlier 3.0 2>&1 | tee -a $L | tail -4
  $V preflight_ram.py "$ART" 2>&1 | grep -q "\-> OK" && $V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m,t = load("$ART")
print("SMOKE OK:", repr(generate(m,t,prompt="The capital of France is", max_tokens=8)))
PY
  $V - <<PY 2>&1 | tee -a $L
import pathlib, mlx.core as mx, math
from mlx_lm.utils import load
p=pathlib.Path("$ART")
print(f"MEASURED SIZE {p.name}: {sum(f.stat().st_size for f in p.glob('*.safetensors'))/2**30:.3f} GiB")
m,t=load("$ART")
ids=t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos=getattr(t,"bos_token_id",None)
if bos is not None and (not ids or ids[0]!=bos): ids=[bos]+ids[:2048]
lg=m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt=mx.array(ids[1:]); lse=mx.logsumexp(lg,axis=-1)
pk=mx.take_along_axis(lg,tgt[:,None].astype(mx.int64),axis=-1)[:,0]
print("PPL", math.exp(float(mx.mean(lse-pk).item())))
PY
  $V kl_damage.py score --model "$ART" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -3
  say "=== ARM $n done"
done
say "E127 ARM C DONE — |A-C| is the noise floor; report A-vs-B against it. Original bar:  — INVERSION if B has lower relerr AND worse ppl/KL; TRACKS if lower relerr AND better; VOID as a specimen if B's relerr is not lower. refs q4 45.842/5.2055; E124 (=arm A geometry) 40.327/5.2330 @13.596 GiB"
