#!/bin/sh
# E95 SCORE — the rebuilt dense 27B (e95-27b-dense-vq-r2, gate PASS 0 outliers)
# on the SAME instrument as the q2/q3/q4 ladder (qwen38_crush.json).
#
# kl_ppl_calibrate.py is NOT invoked whole because it re-scores the bf16
# teacher (55.6G resident) every run, and E115's fit owns the box tonight.
# Instead the two per-rung measurements are executed EXACTLY as its ppl_of()
# and kl_of() do — same corpus, same max-tokens (2048), same cache
# (kl_cache_qwen38) — and merged with the recorded bf16_ppl 5.22490149745195.
# Same instrument; only the redundant teacher reload is skipped.
#
# III.10: FIRST, generate one token through the fused path the artifact ships
# with (bundled model.py via mlx_lm.load). A model that cannot produce a token
# must not produce a ppl.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
ART="$E/e95-27b-dense-vq-r2"
L=logs_live_e95_score.log

echo "########## E95 SCORE $(date '+%H:%M:%S')" | tee -a $L
echo "--- III.10 smoke: one generated token through the shipped runtime" | tee -a $L
$V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m, t = load("$ART")
out = generate(m, t, prompt="The capital of France is", max_tokens=8)
print("SMOKE OK:", repr(out))
PY
grep -q "SMOKE OK" $L || { echo "SMOKE FAILED — no score" | tee -a $L; exit 1; }

echo "--- ppl (referee_corpus, 2048 tok, same math as kl_ppl_calibrate.ppl_of)" | tee -a $L
$V - <<PY 2>&1 | tee -a $L
import mlx.core as mx, math
from mlx_lm.utils import load
m, t = load("$ART")
ids = t.encode(open("referee/referee_corpus.txt").read())[:2048 + 1]
bos = getattr(t, "bos_token_id", None)
if bos is not None and (not ids or ids[0] != bos):
    ids = [bos] + ids[:2048]
lg = m(mx.array([ids[:-1]])).astype(mx.float32)[0]
tgt = mx.array(ids[1:])
lse = mx.logsumexp(lg, axis=-1)
pk = mx.take_along_axis(lg, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
print("PPL", math.exp(float(mx.mean(lse - pk).item())))
PY

echo "--- KL vs bf16 teacher cache (kl_cache_qwen38)" | tee -a $L
$V kl_damage.py score --model "$ART" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -5

echo "########## E95 SCORE DONE $(date '+%H:%M:%S') — refs (same instrument): bf16_ppl 5.2249; q4 45.842mn/89.82%/ppl 5.2055; q3 187.765mn/79.48%/ppl 5.8323; q2 1426.891mn/46.07%/ppl 16.4349. Artifact 9.61 GiB tensors vs q4 14.09 GiB — PLACEMENT reading, not size-matched." | tee -a $L
