#!/bin/sh
# NIGHT STAGE 2 (M3): fill the LADDERS the paper needs, after the headline
# experiments clear. Waits for run_night_final.sh rather than editing it — a
# running chain is never edited (sh reads incrementally).
#
# Two gaps the paper actually has:
#   A. 27B dense ladder. Rungs are fit on M4 but UNPACKED (4.0 bpw regardless
#      of K, III.8) and unscored. Each needs: gate -> pack_dense -> smoke the
#      PACKED artifact (a path verified on e4b but never on qwen3_8) -> score.
#      Processes whatever exists WHEN IT RUNS and LOGS WHAT IT SKIPPED — no
#      silent caps.
#   B. 35B second family. E94's scored artifact was overwritten (retraction in
#      EXPERIMENTS); e94b-35b-K8192-refit-0821 is today's bytes under an
#      honest name, never scored. Scoring it restores a LIVE second-family
#      point under its own name — it does NOT inherit E94's row.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
L=logs_live_night_ladders.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }

while pgrep -f run_night_final2 >/dev/null; do sleep 60; done
say "stage 1 complete — starting ladder fills"

# ---- A. 27B dense rungs
for d in "$E"/e119-27b-dense-*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  case "$n" in *-packed|*-vq) continue;; esac
  # The M4 ships these ALREADY ASSEMBLED (config carries vq_linear, not a
  # fit-only dir), so there is nothing to build — an earlier version of this
  # script would have tried to build an artifact from an artifact. Detect
  # which form is on disk instead of assuming.
  if python3 -c "import json,sys; c=json.load(open('$d/config.json')); sys.exit(0 if c.get('vq_linear') else 1)" 2>/dev/null; then
    ART="$d"; PK="$E/$n-packed"
  else
    ART="$E/$n-vq"; PK="$E/$n-vq-packed"
  fi
  if [ -f "$PK/model.safetensors.index.json" ]; then say "SKIP $n (already packed)"; continue; fi
  say "=== $n  (artifact=$(basename $ART))"
  if [ ! -f "$ART/model.safetensors.index.json" ]; then
    $V build_dense_vq.py --family qwen3_8 --base "$E/qwen38-27b-rungs/q4" \
       --mlp "$d" --out "$ART" >> $L 2>&1 || { say "BUILD FAILED $n"; continue; }
  fi
  $V verify_artifact.py --artifact "$ART" --src "$E/Qwen--Qwen3.8-27B" \
     --family qwen3_8_dense --outlier 3.0 2>&1 | tee -a $L | tail -3
  $V pack_dense.py --src "$ART" --out "$PK" 2>&1 | tee -a $L | tail -2 \
     || { say "PACK FAILED $n"; continue; }
  $V preflight_ram.py "$PK" 2>&1 | tee -a $L | grep -q "\-> OK" && $V - <<PY 2>&1 | tee -a $L
from mlx_lm.utils import load
from mlx_lm import generate
m, t = load("$PK")
print("SMOKE OK:", repr(generate(m, t, prompt="The capital of France is", max_tokens=8)))
PY
  $V - <<PY 2>&1 | tee -a $L
import pathlib
p = pathlib.Path("$PK")
tot = sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"MEASURED SIZE {p.name}: {tot} bytes = {tot/2**30:.3f} GiB")
PY
  $V kl_damage.py score --model "$PK" --cache-dir "$E/kl_cache_qwen38" 2>&1 | tee -a $L | tail -3
  say "=== $n done"
done
say "27B rungs present at run time processed; any rung the M4 finishes AFTER this point is NOT covered here"

# ---- B. 35B second-family point
B="$E/e94b-35b-K8192-refit-0821"
if [ -d "$B" ]; then
  say "=== e94b 35B (own name; does NOT inherit E94's row)"
  $V kl_damage.py score --model "$B" --cache-dir "$E/kl_cache_qwen36" 2>&1 | tee -a $L | tail -4
  $V - <<PY 2>&1 | tee -a $L
import pathlib
p = pathlib.Path("$B")
tot = sum(f.stat().st_size for f in p.glob("*.safetensors"))
print(f"MEASURED SIZE {p.name}: {tot} bytes = {tot/2**30:.3f} GiB")
PY
  say "refs, same instrument: qwen36-35b-rungs/vq-K8192-d4 = 56.413 mnats 89.37% @17.651 GiB"
fi
say "LADDER FILLS COMPLETE"
