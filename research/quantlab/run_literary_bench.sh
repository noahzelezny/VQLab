#!/bin/bash
# Literary-understanding bench across the sidecar candidates.
#
# Same harness discipline as run_task_bench.sh: our scorer, our numbers,
# never mixed with another publisher's. Resumable — a model whose results
# JSON already exists is skipped.
#
# THE SELFTEST GATE IS NOT OPTIONAL (referee/README.md rule). Before any
# litbench number counts, the model must reproduce its known referee ppl
# through score_tasks_streaming.py. Pass SKIP_SELFTEST=1 only for stock
# comparators that have no published referee figure yet.
#
# HF_HOME must be set explicitly: Noah's lives in ~/.zshrc, which
# non-interactive shells do NOT source.
set -u

cd "$(dirname "$0")" || exit 1
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"

MODELS_ROOT="/Volumes/Thunderbay SSD/Exo Models"
OUT="results_literary"
PY="./venv/bin/python"
BATCH="${BATCH:-256}"
SKIP_SELFTEST="${SKIP_SELFTEST:-0}"

mkdir -p "$OUT"

# Rebuild + revalidate the item set every run. It is hand-authored; the
# build is the only thing standing between an authoring slip and a number.
"$PY" literary/build_litbench.py || exit 1
echo

# THE COMPARISON OF RECORD: bf16 vs bf16.
#
# This is the clean form of the capability question — "is a 26B MoE better at
# literary understanding than a 9B dense" — because at bf16 there is NO
# quantization confound on either side. Any community quant would leave the
# result ambiguous between "the model is better" and "their quantizer was
# kinder", and mixing two publishers' quants is what the standing methodology
# rule forbids (score_tasks_streaming.py:23-26).
#
# Memory is not a constraint: the scorer streams one block at a time, flat
# ~15 GB regardless of artifact size, so a 52G bf16 costs no more resident
# than a 9G one.
MODELS=(
  "mlx-community--gemma-4-e4b-it-bf16"        # ~18G, incumbent family at full fat
  "mlx-community--gemma-4-26b-a4b-it-bf16"    # ~52G, the candidate at full fat
)

# SECONDARY, and never to be reported beside the bf16 pair: other publishers'
# quants. Useful only as a sanity check that the bf16 ordering survives
# compression at all. Set INCLUDE_QUANTS=1 to score them too.
if [ "${INCLUDE_QUANTS:-0}" = "1" ]; then
  MODELS+=(
    "mlx-community--gemma-4-e4b-it-8bit"      # 8.4G, the incumbent as deployed
    "mlx-community--gemma-4-26b-a4b-it-4bit"  # 15G
    "mlx-community--Qwen3.6-35B-A3B-4bit"     # 19G, the code-leaning option
  )
fi

echo "=== literary bench: batch=$BATCH ==="
date

# Models live in one of two layouts: a top-level symlink ("$ROOT/<name>") for
# most, or only the HF cache tree ("$ROOT/hub/models--<name>/snapshots/<sha>")
# for anything pulled without one. Resolve both, and require that the
# resolved dir actually holds weights — a snapshot with just config.json is
# the failure mode that looks present and is not.
resolve() {
  local n="$1" p
  for p in "$MODELS_ROOT/$n" "$MODELS_ROOT/hub/models--$n/snapshots"/*; do
    [ -d "$p" ] || continue
    if compgen -G "$p/*.safetensors" >/dev/null; then echo "$p"; return 0; fi
  done
  return 1
}

for m in "${MODELS[@]}"; do
  if ! path="$(resolve "$m")"; then
    echo "SKIP $m — no weights on disk (config-only snapshots do not count)"
    continue
  fi
  if [ -f "$OUT/$m.json" ]; then
    echo "SKIP $m — already scored"
    continue
  fi

  if [ "$SKIP_SELFTEST" != "1" ]; then
    echo "--- selftest $m"
    "$PY" score_tasks_streaming.py --model "$path" --selftest \
      --batch-seqs "$BATCH" || { echo "SELFTEST FAILED $m — skipping"; continue; }
  fi

  echo "--- litbench $m"
  "$PY" score_tasks_streaming.py \
    --model "$path" \
    --tasks litbench \
    --include-path literary \
    --batch-seqs "$BATCH" \
    --output-dir "$OUT" || echo "FAILED $m"
done

date
echo "=== done. analyze with: ./analyze_task_bench.py --dir $OUT --tasks litbench"
