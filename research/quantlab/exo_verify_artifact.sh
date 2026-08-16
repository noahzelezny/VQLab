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

echo "=== [$(date +%H:%M:%S)] clearing previous instances ==="
for iid in $(curl -s -m 5 "$EP/state" | python3 -c "
import json,sys
try: print(' '.join(json.load(sys.stdin).get('instances',{}).keys()))
except Exception: pass"); do
  curl -s -X DELETE "$EP/instances/$iid" >/dev/null 2>&1 && echo "  deleted $iid"
done
sleep 5

echo "=== [$(date +%H:%M:%S)] placing $MODEL (2 nodes, tensor) ==="
t0=$(date +%s)
python3 "$ROOT/$Q" --model "$MODEL" --min-nodes 2 --wait 1500 2>&1 | tail -5
rc=$?
t1=$(date +%s)
echo "  placement returned rc=$rc after $((t1-t0))s"
[ $rc -ne 0 ] && { echo "VERDICT: $MODEL FAILED TO PLACE"; exit 1; }

echo "=== [$(date +%H:%M:%S)] generating (coherence check) ==="
resp=$(curl -s -m 300 "$EP/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\":\"user\",\"content\":\"In two sentences, what is vector quantization?\"}],
  \"max_tokens\": 160, \"temperature\": 0.6}")
echo "$resp" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print('  !! non-JSON response:', sys.stdin.read()[:200]); sys.exit(1)
if 'error' in d: print('  !! error:', json.dumps(d['error'])[:300]); sys.exit(1)
txt=d['choices'][0]['message']['content']
u=d.get('usage',{})
print('  --- MODEL OUTPUT ---')
for line in txt.strip().splitlines(): print('  |', line)
print('  --- tokens:', u.get('completion_tokens'), '---')
sys.exit(0 if txt.strip() else 1)"
gen=$?
[ $gen -ne 0 ] && { echo "VERDICT: $MODEL PLACED BUT DID NOT GENERATE"; exit 1; }
echo "VERDICT: $MODEL SERVES — read the output above and judge coherence."
