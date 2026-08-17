#!/bin/bash
# RESIDENT prefill probes — run on the M4, off the M3's disk-bound path.
#
# WHY. The 08-17 chunk sweep concluded "no knee" from score_tasks_streaming
# --selftest, which RE-READS the whole model from disk every pass (~63 s of a
# ~100 s run for 100 GiB at ~1.7 GB/s). That measurement is DISK-BOUND, so a
# compute-side change is largely invisible in it — the VQ-PF1 probe measured
# count-sorted _prefill at 75.9 ms @ chunk16 vs 124.7 ms @ chunk128 (1.64x),
# which the streaming sweep reported as "no trend". Both can be true. This
# script uses probe_block_prefill.py, which loads ONE block and times
# blk(h) with the weights RESIDENT — no disk in the timed region.
#
# TEST A — is there a chunk knee for RESIDENT inference?
# TEST B — does the gap depend on WORKLOAD SHAPE? The regime table says our
#   kernel is at parity at 177 rows/expert and WINS at 354 (1.20x) and 1414
#   (1.60x). rows/expert = tokens*top_k/512, so a long prompt should land in
#   the winning regime while the benchmark's short padded buckets do not.
#   If VQ is already at/above parity at 30k-context shapes, the "4-5x gap"
#   is a benchmark artifact, not a property users experience.
#
# CAVEAT carried from the VQ-PF1 entry: the probe drives the block with
# RANDOM hidden states. Real routing skew was measured at 8.7x on random
# states too, so the shape is faithful "in kind, not necessarily magnitude" —
# do not quote these as user-facing throughput, only as A/B ratios.
set -u
cd ~/quantlab || exit 1
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"

VQ="$HOME/.exo/models/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw"
SPICY="/Volumes/Thunderbay SSD/Exo Models/spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit"

echo "############ TEST A — chunk knee, RESIDENT, benchmark shape [256,36]"
for c in 8 16 32 64 128; do
  echo ""
  echo "--- SCOUT_VQ_DECODE_CHUNK=$c"
  SCOUT_VQ_DECODE_CHUNK=$c venv/bin/python probe_block_prefill.py \
    --model "$VQ" --layer 3 --batch-seqs 256 --seq-len 36 --buckets 3 \
    2>/dev/null | grep -E "steady-state|bucket [0-9]"
done

echo ""
echo "############ TEST B — workload shape: VQ vs spicy at increasing tokens/call"
# tokens/call -> rows/expert = tokens*10/512
#   9216 (benchmark)  -> 180     8192  -> 160
#  16384              -> 320    32768  -> 640    65536 -> 1280
for shape in "256 36" "1 8192" "2 8192" "4 8192" "8 8192"; do
  set -- $shape; B=$1; L=$2
  echo ""
  echo "=== bucket [$B, $L] = $((B*L)) tokens  (~$((B*L*10/512)) rows/expert)"
  for m in "$VQ" "$SPICY"; do
    printf "  %-46s " "$(basename "$m")"
    venv/bin/python probe_block_prefill.py --model "$m" --layer 3 \
      --batch-seqs "$B" --seq-len "$L" --buckets 3 2>/dev/null \
      | grep "steady-state" | sed 's/.*per bucket: //'
  done
done
echo ""
echo "############ done"
date
