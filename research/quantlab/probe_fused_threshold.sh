#!/bin/bash
# THE INVISIBLE LEVER: is VQ_FUSED_MAX_N set on the wrong side of the default
# prefill step?
#
# mlx-lm prefills in 512-token steps by default. top_k=10, so each call is
# N = 512*10 = 5,120 (token,expert) pairs. VQ_FUSED_MAX_N is 4096 — so 5,120
# JUST clears it and takes the chunked padded-GEMM path at ~10 rows/expert,
# which the regime table calls our worst case (0.33x vs gather_qmm at 20
# rows/expert). The fused kernel never gets a shot at that shape.
#
# If fused is faster there, raising the threshold is a change INSIDE model.py:
# every user gets it with no env var, no flag, no intervention. That is worth
# more than a tuning note, per Noah — invisible beats documented.
#
# Runs on the M3 despite the model not fitting in 96 GB, because
# probe_block_prefill materializes ONE block (~1.7 GiB). Different instrument
# from the M4 run (which drives full generation via m2_speed_split), so the
# two are an independent cross-check rather than a repeat.
#
# Shapes chosen to bracket the threshold:
#   [1, 256]  -> N=2,560  already fused under both settings (control)
#   [1, 512]  -> N=5,120  THE DEFAULT PREFILL STEP — the case that matters
#   [1, 1024] -> N=10,240
#   [1, 2048] -> N=20,480
set -u
cd "$(dirname "$0")" || exit 1
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"
M="/Volumes/Thunderbay SSD/Exo Models/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw"

echo "=== fused-threshold probe (M3, resident single block, chunk 32)"
date
for L in 256 512 1024 2048; do
  N=$((L*10))
  echo ""
  echo "### bucket [1, $L]  N=$N pairs  (~$((N/512)) rows/expert)"
  for f in 4096 65536; do
    label="padded-GEMM"; [ "$f" -gt "$N" ] && label="FUSED kernel"
    for r in 1 2 3; do
      printf "  FUSED_MAX_N %6s run%s  %-12s " "$f" "$r" "$label"
      SCOUT_VQ_DECODE_CHUNK=32 SCOUT_VQ_FUSED_MAX_N=$f \
        venv/bin/python probe_block_prefill.py --model "$M" --layer 3 \
        --batch-seqs 1 --seq-len "$L" --buckets 4 2>/dev/null \
        | grep "steady-state" | sed 's/.*per bucket: //' || echo FAILED
    done
  done
done
echo ""
echo "=== done — compare the two FUSED_MAX_N blocks at each shape."
echo "=== N below 4096 should be IDENTICAL (both fused); the interesting"
echo "=== rows are N=5120 and up, where 4096 takes the padded path."
date
