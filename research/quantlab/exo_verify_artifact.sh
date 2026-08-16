#!/usr/bin/env bash
# exo_verify_artifact.sh — prove ONE artifact actually serves on the 2-node ring.
#
# Publishing gate, not a smoke test: a model that loads but emits garbage is
# worse than one that fails to load, because the failure is silent. So this
# places it, generates real text, and prints the text for a human to read.
#
#   ./exo_verify_artifact.sh TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw
set -u
MODEL="${1:?usage: $0 <model_id>}"
EP="http://127.0.0.1:52415"
Q="scripts/exo_place_wait.py"
ROOT="/Users/noahzelezny/Documents/AgenicAI"

# FULL RESET, not DELETE. Deleting an instance leaves runners stuck in
# RunnerShuttingDown forever and any subsequent placement spawns runners that
# die instantly, leaving ghost "RunnerLoading" entries with NO process behind
# them (observed 2026-08-16 on the 2.4bpw run — the state API reports progress
# that is pure fiction). A clean reset is the only reliable way to sequence
# two placements. Costs ~60s; buys a trustworthy test.
echo "=== [$(date +%H:%M:%S)] full exo reset (clean slate) ==="
bash "$ROOT/scripts/exo-reset.sh" --restart 2>&1 | tail -3
sleep 8

echo "=== [$(date +%H:%M:%S)] placing $MODEL (2 nodes, tensor) ==="
t0=$(date +%s)
python3 "$ROOT/$Q" --model "$MODEL" --min-nodes 2 --wait 1500 2>&1 | tail -5
rc=$?
t1=$(date +%s)
echo "  placement returned rc=$rc after $((t1-t0))s"
[ $rc -ne 0 ] && { echo "VERDICT: $MODEL FAILED TO PLACE"; exit 1; }

echo "=== [$(date +%H:%M:%S)] generating (coherence check) ==="
# NOTE: thinking model — it reasons regardless of /no_think, so the budget must
# be large enough to finish or the visible answer truncates to a stray char and
# LOOKS like corruption. 3000 tokens is comfortable for a trivial prompt.
resp=$(curl -s -m 300 "$EP/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\":\"user\",\"content\":\"In two sentences, what is vector quantization?\"}],
  \"max_tokens\": 3000, \"temperature\": 0.6}")
echo "$resp" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print('  !! non-JSON response:', sys.stdin.read()[:200]); sys.exit(1)
if 'error' in d: print('  !! error:', json.dumps(d['error'])[:300]); sys.exit(1)
m=d['choices'][0]['message']
txt=(m.get('content') or '').strip()
think=(m.get('reasoning_content') or '').strip()
u=d.get('usage',{}); fin=d['choices'][0].get('finish_reason')
print('  --- MODEL OUTPUT ---')
for line in (txt or '(empty)').splitlines(): print('  |', line)
print('  --- tokens:', u.get('completion_tokens'), ' finish:', fin, ' reasoning_chars:', len(think), '---')
if fin == 'length':
    print('  !! TRUNCATED — raise max_tokens; a thinking model needs room to finish')
    sys.exit(1)
sys.exit(0 if txt else 1)"
gen=$?
[ $gen -ne 0 ] && { echo "VERDICT: $MODEL PLACED BUT DID NOT GENERATE"; exit 1; }
echo "VERDICT: $MODEL SERVES — read the output above and judge coherence."
