#!/bin/sh
# Single follow-on waiter: E47-verify the 397B session's gemma cheap-shallow
# artifact once the E56 benches release the machine. One waiter only —
# do not add siblings (stampede rule).
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
while pgrep -f "gemma_small_verdict.sh" >/dev/null; do sleep 120; done
./venv/bin/python verify_artifact.py \
  --artifact "/Volumes/Thunderbay SSD/Exo Models/gemma26b-rungs/vq-headdown-k128-tail512-d4" \
  --src "$(echo "/Volumes/Thunderbay SSD/Mlx_Models/hub/models--mlx-community--gemma-4-26b-a4b-it-bf16/snapshots"/*)" \
  --family gemma4 --outlier 3.0
echo "NOTE: mixed-geometry build (K128 shallow / K512 tail) — read the gate"
echo "per-region per the E57 amendment; shallow K128 relerr WILL exceed the"
echo "K512 body median legitimately."
