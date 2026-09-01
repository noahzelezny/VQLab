#!/bin/bash
# VQ-PF1 follow-up: is there a KNEE in _DECODE_CHUNK that buys prefill speed
# at an acceptable perplexity cost?
#
# WHAT WE KNOW GOING IN (all n=1, which is why this sweep exists):
#   chunk 128 (shipped, count-sorted) -> 2.2bpw selftest 91.4 s, ppl 3.1706
#                                        == the published number, exactly
#   chunk 16                          -> 76.3 s, ppl 3.1754  (+0.0048)
# The ppl shift is NOT quantization damage: `ne` is the batched-GEMM batch
# dimension, a different ne selects a different Metal tiling, and fp16
# accumulates in a different order. That predicts something testable:
#   - ARITHMETIC NOISE  => ppl scatters around 3.1706, some chunks BELOW it
#   - SYSTEMATIC        => ppl climbs monotonically as chunk shrinks
# Those imply different answers to "is there an acceptable knee", and one
# data point cannot tell them apart.
#
# ppl is DETERMINISTIC and contention-immune; wall time is not. Run the
# timing half on a quiet box or treat times as upper bounds.
#
# Each run re-streams the whole model (the scorer frees each block), so cost
# is ~1.5-3 min per point regardless of how few tokens are scored.
#
#   ./sweep_chunk_knee.sh [model_dir] [chunks...]
set -u
cd "$(dirname "$0")" || exit 1
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"

MODEL="${1:-/Volumes/Thunderbay SSD/Exo Models/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw}"
shift 2>/dev/null || true
CHUNKS=("${@:-4 8 16 32 64 128}")
[ $# -gt 0 ] && CHUNKS=("$@")

NAME="$(basename "$MODEL")"
OUT="logs_chunk_knee_${NAME}.tsv"
echo -e "chunk\tppl\ttotal_nll\tseconds" > "$OUT"

echo "=== chunk knee sweep: $NAME"
echo "=== published ppl for reference: 2.2bpw 3.1706 | 2.4bpw 2.7655 | 3.1bpw 2.3519"
date

for c in ${CHUNKS[@]}; do
  echo ""
  echo "--- SCOUT_VQ_DECODE_CHUNK=$c ($(date +%H:%M:%S))"
  line=$(SCOUT_VQ_DECODE_CHUNK="$c" venv/bin/python score_tasks_streaming.py \
           --model "$MODEL" --selftest --batch-seqs 64 2>/dev/null \
         | python3 -c "
import json,sys
t=sys.stdin.read()
i=t.find('{'); j=t.rfind('}')
d=json.loads(t[i:j+1])
print('%s\t%s\t%s'%(d['ppl'], d['total_nll'], d['seconds']))
" 2>/dev/null)
  if [ -n "$line" ]; then
    echo -e "$c\t$line" | tee -a "$OUT"
  else
    echo -e "$c\tFAILED" | tee -a "$OUT"
  fi
done

echo ""
echo "=== results ($OUT):"
column -t "$OUT"
echo ""
echo "READ IT THIS WAY: if ppl is monotone in chunk, the effect is systematic"
echo "and a knee is a real quality/speed trade. If it scatters (some chunks"
echo "at or below the published ppl), it is float ordering and the 'cost' of"
echo "a smaller chunk is not a cost at all -- but it still breaks bit-exact"
echo "reproduction of the published numbers, which is a separate decision."
date
