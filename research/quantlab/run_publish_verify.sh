#!/bin/sh
# PUBLISH VERIFICATION — the two rungs Noah intends to swap (08-21).
#
# 101 GiB: shipped 2.2 (3.1706)      -> d8K16384-packed      (3.0591/2.6728)
# 144 GiB: shipped 3.1 (2.3519/2.5987) -> flatk2048-refit-packed (2.3410/2.5963)
#
# WHY THIS EXISTS. d8K16384-packed was SCORED at 11:39 with one runtime and
# its model.py was REPLACED at 16:29 with the 1208-line bundle carrying the
# packed d8 fused kernel. So the artifact currently ships a decode path that
# has never produced the published number, and fused-vs-decode is a numerics
# change, not a packaging one. E81 in its published form. We re-score THE
# BYTES WE INTEND TO PUBLISH with THE RUNTIME WE INTEND TO PUBLISH.
# flatk2048-refit-packed does not have this problem (model.py 03:14, scored
# 03:17) but is smoked and release-checked here anyway — it costs seconds and
# the whole point is that the shipping bundle is what gets verified.
#
# III.11 smoke FIRST on each: an artifact that cannot generate a token must
# not produce a ppl.
#
# READING: the d8 re-score either reproduces ~3.0591/2.6728 or it does not.
#   REPRODUCES (within 0.005 both corpora) -> the runtime swap is numerically
#     inert; the published number describes the shipped bytes. Clear to card.
#   DOES NOT -> the fused kernel changes outputs. The 11:39 number is then
#     void for publication and THIS run's number is the artifact's number.
#     Do not average, do not pick the better one, do not publish until Noah
#     has seen both.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
L=logs_live_publish_verify.log

check() {
  A="$1"
  echo "########## $A" | tee -a $L
  md5 -q "$E/$A/model.py" | sed 's/^/  bundled runtime md5: /' | tee -a $L
  wc -l < "$E/$A/model.py" | sed 's/^/  bundled runtime lines: /' | tee -a $L
  echo "--- III.11 smoke: one token through the SHIPPED bundle" | tee -a $L
  $V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m, t = load("$E/$A")
print("SMOKE OK:", repr(generate(m, t, prompt="The capital of France is", max_tokens=8)))
PY
  $V check_release.py --artifact "$E/$A" 2>&1 | tee -a $L | tail -1
  $V - <<PY 2>&1 | tee -a $L
import pathlib
p = pathlib.Path("$E/$A")
tot = sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"SIZE whole-artifact: {tot} bytes = {tot/2**30:.3f} GiB")
PY
}

echo "########## PUBLISH VERIFY $(date '+%H:%M:%S')" | tee -a $L
check rotlab--397B-d8K16384-packed
echo "--- RE-SCORE with the shipping runtime (was scored 11:39, runtime replaced 16:29)" | tee -a $L
$V referee/score_streaming.py --model "$E/rotlab--397B-d8K16384-packed" --corpus referee/referee_corpus.txt 2>&1 | tee -a $L | tail -1
$V referee/score_streaming.py --model "$E/rotlab--397B-d8K16384-packed" --corpus referee/referee_corpus_code.txt 2>&1 | tee -a $L | tail -1

check rotlab--397B-flatk2048-refit-packed

echo "########## PUBLISH VERIFY DONE $(date '+%H:%M:%S')" | tee -a $L
echo "  d8 refs (11:39, OLD runtime): 3.0591 wikitext / 2.6728 code @100.971 GiB; incumbent shipped 2.2 = 3.1706 @100.930" | tee -a $L
echo "  k2048 refs (03:17, same runtime): 2.3410 / 2.5963 @143.682 GiB; incumbent shipped 3.1 = 2.3519 / 2.5987 @143.682" | tee -a $L
