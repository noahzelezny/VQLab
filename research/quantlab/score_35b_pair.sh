#!/bin/sh
# Score the two 35B rungs on ONE instrument (kl_cache_qwen36 — teacher
# Qwen3.6-35B-A3B-bf16, the same cache that produced e94b's standing 53.022,
# per run_night_ladders.sh:71). Comparator first, then R2, so R2's verdict
# faces a number measured in the same session (III.3).
#
# DEVIATION FROM E132, STATED NOT HIDDEN: E132 registered "gate or smoke fails
# -> stop." The III.11 smoke FAILS on both of these and the cause is known and
# artifact-independent (E134: d4 K>=4096 MoE fused kernels exceed Apple's 32 KB
# threadgroup cap; confirmed on BOTH boxes). E132's stop clause was written to
# prevent papering over a BROKEN COMPARATOR; here comparator and subject fail
# identically for a runtime reason. So these numbers are QUALITY BETWEEN TWO
# UNRELEASABLE ARTIFACTS and must be labelled that way. Neither is an offering
# until E134 is fixed. Noah directed these runs after E134 was confirmed.
# No smoke is attempted: already run on e94b-packed (E134) and it failed.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
CACHE="$E/kl_cache_qwen36"
L=logs_live_35b_pair.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }
for A in e94b-35b-K8192-refit-0821-packed e128-35b-d4K16384-packed; do
  S="$E/$A"
  [ -d "$S" ] || { say "MISSING $A"; continue; }
  say "=== $A"
  $V preflight_ram.py "$S" 2>&1 | tee -a $L | tail -1
  $V - <<PY 2>&1 | tee -a $L
import pathlib
p=pathlib.Path("$S")
print(f"MEASURED SIZE {p.name}: {sum(f.stat().st_size for f in p.glob('*.safetensors'))/2**30:.3f} GiB")
PY
  $V - <<PY 2>&1 | tee -a $L
import pathlib, mlx.core as mx, math
from mlx_lm.utils import load
m,t=load("$S")
ids=t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos=getattr(t,"bos_token_id",None)
if bos is not None and (not ids or ids[0]!=bos): ids=[bos]+ids[:2048]
lg=m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt=mx.array(ids[1:]); lse=mx.logsumexp(lg,axis=-1)
pk=mx.take_along_axis(lg,tgt[:,None].astype(mx.int64),axis=-1)[:,0]
print("PPL", math.exp(float(mx.mean(lse-pk).item())))
PY
  $V kl_damage.py score --model "$S" --cache-dir "$CACHE" 2>&1 | tee -a $L | tail -3
  say "=== $A done"
done
say "35B PAIR COMPLETE — refs same instrument: q8 7.449, e94b(unpacked) 53.022, q4 78.557"
