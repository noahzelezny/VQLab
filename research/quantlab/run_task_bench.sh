#!/bin/bash
# Task-benchmark sweep: HellaSwag / PIQA / WinoGrande across the release
# lineup AND the two comparators, all on OUR harness.
#
# Ordered so the THREE ARTIFACTS WE PUBLISH finish first: if the run is
# interrupted, the deliverable exists and only the comparators are missing.
#
# Resumable by design -- a model whose results JSON already exists is
# skipped, so re-running after a failure costs nothing. Each model is one
# process and one streamed pass over its bytes (lm-eval batches all three
# tasks' requests into a single loglikelihood call).
#
# HF_HOME must be set explicitly: Noah's lives in ~/.zshrc, which
# non-interactive shells do NOT source, and without it dataset access runs
# unauthenticated.
set -u

cd "$(dirname "$0")" || exit 1
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"

MODELS_ROOT="/Volumes/Thunderbay SSD/Exo Models"
OUT="results_tasks"
LIMIT="${LIMIT:-1000}"
SHOTS="${SHOTS:-0}"
BATCH="${BATCH:-256}"

mkdir -p "$OUT"

# ours first, then the comparators
MODELS=(
  "Qwen3.5-397B-A17B-VQ-2.2bpw"
  "Qwen3.5-397B-A17B-VQ-2.4bpw"
  "Qwen3.5-397B-A17B-VQ-3.1bpw"
  "spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit"
  "spicyneuron--Qwen3.5-397B-A17B-MLX-3.5bit"
)

echo "=== task bench: limit=$LIMIT shots=$SHOTS batch=$BATCH ==="
date

for m in "${MODELS[@]}"; do
  if [ -f "$OUT/$m.json" ]; then
    echo "[skip] $m (results already present)"
    continue
  fi
  if [ ! -d "$MODELS_ROOT/$m" ]; then
    echo "[MISSING] $m -- not on disk, skipping"
    continue
  fi
  echo ""
  echo "=== $m  ($(date +%H:%M:%S)) ==="
  venv/bin/python score_tasks_streaming.py \
    --model "$MODELS_ROOT/$m" \
    --tasks hellaswag,piqa,winogrande \
    --limit "$LIMIT" --num-shots "$SHOTS" \
    --batch-seqs "$BATCH" \
    --output-dir "$OUT"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL rc=$rc] $m -- continuing; re-run this script to retry"
  fi
done

echo ""
echo "=== sweep done ==="
date
ls -la "$OUT"
