#!/bin/sh
# Close the ppl gap FINDINGS 6c opens: the d4 rungs were scored on KL only,
# and KL has now demonstrably disagreed with ppl (E124). Any rung ranked
# against q3/q4 on KL alone is ranked on a metric we know can invert.
# Same ppl method as E95/E124 (referee_corpus, 2048 tok, bf16_ppl 5.2249).
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
L=logs_live_ppl_fill.log
for A in e119-27b-dense-k512-packed e119-27b-dense-k1024-packed e95-27b-dense-vq-r2; do
  [ -d "$E/$A" ] || { echo "MISSING $A" | tee -a $L; continue; }
  echo "--- $A" | tee -a $L
  $V - <<PY 2>&1 | tee -a $L
import mlx.core as mx, math
from mlx_lm.utils import load
m, t = load("$E/$A")
ids = t.encode(open("referee/referee_corpus.txt").read())[:2049]
bos = getattr(t, "bos_token_id", None)
if bos is not None and (not ids or ids[0] != bos): ids = [bos] + ids[:2048]
lg = m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt = mx.array(ids[1:]); lse = mx.logsumexp(lg, axis=-1)
pk = mx.take_along_axis(lg, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
print("PPL", math.exp(float(mx.mean(lse - pk).item())))
PY
done
echo "ppl fill done $(date '+%H:%M:%S') — refs: bf16 5.2249, q4 5.2055, q3 5.8323, q2 16.4349, E124 5.2330" | tee -a $L
