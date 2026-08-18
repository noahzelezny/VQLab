#!/bin/bash
# Overnight crush: Qwen3.8-27B mixed-precision sweep + gemma-4-26b first ladder.
#
# SEQUENTIAL BY NECESSITY. This is the M3 Ultra 96GB (not the M4 128GB, and
# no ssh path to it), and the two bf16 teachers are 55.6G + 48G — they cannot
# both be resident. Every phase is resumable: an artifact or result file that
# already exists is skipped, so a re-run after a failure costs nothing.
#
# ORDERED BY VALUE. Noah asked for the Qwen crush; that runs first and
# completely, so an interruption still leaves the deliverable. Gemma follows.
set -u
cd "$(dirname "$0")" || exit 1
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"

PY=./venv/bin/python
EXO="/Volumes/Thunderbay SSD/Exo Models"
QSRC="$EXO/Qwen--Qwen3.8-27B"
QRUNG="$EXO/qwen38-27b-rungs"
QCACHE="$EXO/kl_cache_qwen38"
GSRC=$(ls -d "$EXO/../Mlx_Models/hub/models--mlx-community--gemma-4-26b-a4b-it-bf16/snapshots/"*/ 2>/dev/null | head -1)
GRUNG="$EXO/gemma26b-rungs"
GCACHE="$EXO/kl_cache_gemma26b"

mkdir -p "$QRUNG" "$GRUNG" results_crush
say() { echo; echo "############ $* ($(date +%H:%M:%S))"; }

# ---------------------------------------------------------------- PHASE A
# Qwen mixed rungs. Budget target ~11G to go head-to-head with uniform q3
# (11G, ppl 1.116x), plus a ~13.5G point against q4 (14G, 0.996x).
# E8's lesson is encoded as an explicit attention floor in every rung.
say "PHASE A: Qwen3.8-27B mixed rungs"
#          name         mlp  mlpgs  linattn  fullattn  embed
RUNGS=(
  "m2-a4          2    64     4        6        4"
  "m2-a6          2    64     6        6        4"
  "m2-a4-gs32     2    32     4        6        4"
  "m3-a4          3    64     4        6        4"
  "m3-a3          3    64     3        6        3"
)
for spec in "${RUNGS[@]}"; do
  set -- $spec
  n=$1; mb=$2; mgs=$3; la=$4; fa=$5; eb=$6
  if [ -d "$QRUNG/$n" ]; then echo "skip $n (exists)"; continue; fi
  echo "--- building $n: mlp${mb}(gs$mgs) linattn${la} fullattn${fa} embed${eb}"
  $PY convert_qwen38_mixed.py --src "$QSRC" --out-root "$QRUNG" --name "$n" \
      --mlp-bits "$mb" --mlp-group-size "$mgs" --linattn-bits "$la" \
      --fullattn-bits "$fa" --embed-bits "$eb" \
    && echo "    -> $(du -shL "$QRUNG/$n" | cut -f1)" \
    || echo "    FAILED $n"
done

say "PHASE B: score every Qwen rung (ppl + KL)"
ALL=()
for d in "$QRUNG"/*; do [ -d "$d" ] && ALL+=("$d"); done
$PY kl_ppl_calibrate.py --teacher "$QSRC" --rungs "${ALL[@]}" \
    --cache-dir "$QCACHE" --corpus referee/referee_corpus.txt \
    --max-tokens 2048 --out results_crush/qwen38_crush.json \
  || echo "PHASE B FAILED"

# ---------------------------------------------------------------- PHASE C
say "PHASE C: gemma-4-26b bf16 teacher cache"
if [ -f "$GCACHE/meta.json" ]; then
  echo "skip (cache exists)"
elif [ -z "$GSRC" ]; then
  echo "SKIP: gemma bf16 snapshot not found"
else
  # chat-wrapped (the sidecar's real regime) + literary corpus, per
  # kl_damage.py's default rationale.
  $PY kl_damage.py cache --model "$GSRC" --out-dir "$GCACHE" \
      --corpus referee/referee_corpus_literary.txt \
      --num-samples 24 --seq-len 512 --batch-size 2 --top-k 64 \
    || echo "PHASE C FAILED"
fi

# ---------------------------------------------------------------- PHASE D
# FIRST REAL TEST of convert_gemma_struct.py. Its module re-targeting
# (switch_glu not switch_mlp, router.proj not mlp.gate) has only ever been
# checked against mlx_lm SOURCE, never against real gemma tensors. The bit
# histogram it prints is the verification: if the expert bucket is empty,
# the predicate is wrong and every later rung would be silently mis-built.
say "PHASE D: gemma-4-26b struct rungs"
if [ -z "$GSRC" ]; then
  echo "SKIP: no gemma bf16"
else
  for spec in "struct6-e2 2" "struct6-e3 3"; do
    set -- $spec
    n=$1; eb=$2
    if [ -d "$GRUNG/rotlab-gemma26B-$n" ]; then echo "skip $n"; continue; fi
    echo "--- building $n (experts ${eb}-bit, structure 6-bit)"
    $PY convert_gemma_struct.py --src "$GSRC" --out-root "$GRUNG" \
        --name "$n" --expert-bits "$eb" \
      && echo "    -> $(du -shL "$GRUNG/rotlab-gemma26B-$n" | cut -f1)" \
      || echo "    FAILED $n"
  done

  say "PHASE D2: score gemma rungs (KL only — ppl is INVALID on gemma-4)"
  for d in "$GRUNG"/*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -f "results_crush/gemma_$b.json" ] && { echo "skip $b"; continue; }
    $PY kl_damage.py score --model "$d" --cache-dir "$GCACHE" \
        --out "results_crush/gemma_$b.json" || echo "FAILED $b"
  done
fi

say "ALL PHASES DONE"
echo "results in results_crush/"
