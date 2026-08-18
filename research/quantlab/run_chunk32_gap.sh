#!/bin/bash
# Clean post-fix gap measurement, all on ONE quiet box, back to back.
#
# WHY THIS RE-RUN EXISTS. Two numbers from 08-16 cannot carry the claim:
#   - our per-artifact times were taken under DIFFERENT contention (the HF
#     upload saturated the SSD until ~19:38, so 2.2bpw ran contended and
#     3.1bpw ran clean -- which is why the LARGEST artifact looked FASTEST,
#     1.28 h vs 1.43 h; that is the disk, not the model).
#   - the "4-5x remaining gap vs spicy" was ARITHMETIC (pre-fix time / 1.83),
#     never a measurement.
# Same box, same hour, same workload, nothing else running => a ratio that
# means something.
#
# Also a free correctness check: count-sorted chunking is numerically exact at
# the shipped chunk, so every task accuracy MUST come back identical to the
# 08-16 run. A drift means the fix is not what we think it is.
#
# Order: spicy first (10 min, gives an early comparator anchor), then 2.2bpw
# (the artifact people actually download), then 2.4bpw (Noah's daily driver).
set -u
export SCOUT_VQ_DECODE_CHUNK=32
cd "$(dirname "$0")" || exit 1
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"

ROOT="/Volumes/Thunderbay SSD/Exo Models"
OUT="results_tasks_chunk32"
mkdir -p "$OUT"

MODELS=(
  "spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit"
  "TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw"
  "TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
)

echo "=== post-fix gap measurement (lm-eval 0.4.12, 0-shot, limit 1000)"
date
for m in "${MODELS[@]}"; do
  if [ -f "$OUT/$m.json" ]; then echo "[skip] $m"; continue; fi
  [ -d "$ROOT/$m" ] || { echo "[MISSING] $m"; continue; }
  echo ""
  echo "=== $m  ($(date +%H:%M:%S))"
  venv/bin/python score_tasks_streaming.py \
    --model "$ROOT/$m" \
    --tasks hellaswag,piqa,winogrande \
    --limit 1000 --num-shots 0 --batch-seqs 256 \
    --output-dir "$OUT" || echo "[FAIL] $m — rerun this script to retry"
done

echo ""
echo "=== POST-FIX RESULTS"
venv/bin/python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("results_tasks_chunk32/*.json")):
    if "samples" in f: continue
    d = json.load(open(f)); r = d["results"]
    rows.append((d["model"], d["seconds"]/3600,
                 r["hellaswag"]["acc_norm,none"], r["piqa"]["acc_norm,none"],
                 r["winogrande"]["acc,none"]))
print("%-46s %6s %9s %7s %8s" % ("model","hrs","hellasw","piqa","winogr"))
for m,h,a,b,c in rows:
    print("%-46s %6.2f %9.4f %7.4f %8.4f" % (m[:46],h,a,b,c))
# ratio vs the spicy comparator measured in the SAME session
spicy = next((h for m,h,_,_,_ in rows if m.startswith("spicyneuron")), None)
if spicy:
    print()
    for m,h,_,_,_ in rows:
        if not m.startswith("spicyneuron"):
            print("  %-44s %.2fx spicy (measured, not derived)" % (m[:44], h/spicy))
# correctness: accuracies must match the 08-16 pre-fix run exactly
print()
print("vs 08-16 (pre-fix) — accuracies MUST be identical:")
for f in sorted(glob.glob("results_tasks_chunk32/*.json")):
    if "samples" in f: continue
    old = os.path.join("results_tasks", os.path.basename(f).replace("TheDrainFlorist--",""))
    if not os.path.exists(old):
        old = os.path.join("results_tasks", os.path.basename(f))
    if not os.path.exists(old):
        print("  (no pre-fix twin for %s)" % os.path.basename(f)); continue
    n = json.load(open(f))["results"]; o = json.load(open(old))["results"]
    same = all(abs(n[t][k]-o[t][k]) < 1e-12
               for t in n for k in n[t] if k.endswith(",none") and k in o[t])
    print("  %-46s %s" % (os.path.basename(f), "IDENTICAL" if same else "*** DRIFT ***"))
PY
date
