#!/bin/sh
# exo_probe_known_answers.sh — follow-up probes against an ALREADY-PLACED exo
# instance. Run after exo_verify_artifact.sh; no reset, no reload.
#
# WHY THIS EXISTS. exo_verify_artifact.sh asks one open-ended question ("what
# is vector quantization?"). That catches fluent garbage but not subtler
# degradation, and it cannot be checked mechanically. These probes are:
#   - GREEDY (temperature 0) so a weird answer cannot be blamed on sampling
#     and the run is reproducible;
#   - graded by difficulty, because failure MODE is the signal:
#       overdetermined -> a badly degraded model still passes (weak bar)
#       two-hop arithmetic + precise recall -> partial degradation fails HERE
#     Fluent garbage fails all three. Partial degradation typically passes the
#     first and fails the last two. ONE prompt cannot distinguish those.
#
# WHAT A PASS DOES NOT PROVE: that exo's sharded output equals a single-box
# run of the same artifact. For a 143.70 GiB model that comparison is
# impossible — it fits neither box. The strongest claim available from this is
# "it serves and is coherent", NEVER "sharding is bit-exact".
set -u
MODEL="${1:?usage: $0 <model_id>}"
EP="http://127.0.0.1:52415"

ask() {
  label="$1"; q="$2"
  printf '\n--- %s\n    Q: %s\n' "$label" "$q"
  curl -s -m 300 "$EP/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],
         \"max_tokens\":3000,\"temperature\":0}" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d: print('    !! error:', json.dumps(d['error'])[:300]); sys.exit(0)
c=d['choices'][0]; m=c['message']
txt=(m.get('content') or '').strip()
for line in (txt or '(empty)').splitlines(): print('    |', line)
print('    [tokens', d.get('usage',{}).get('completion_tokens'),
      ' finish', c.get('finish_reason'),
      ' reasoning_chars', len((m.get('reasoning_content') or '')), ']')
"
}

echo "===== known-answer probes, greedy (temperature 0) — $MODEL"
ask "1 OVERDETERMINED (weak bar; a degraded model still passes)" \
    "The capital of France is"
ask "2 TWO-HOP ARITHMETIC (partial degradation shows here)" \
    "What is 17 times 23? Answer with just the number."
ask "3 PRECISE RECALL (partial degradation shows here)" \
    "Who wrote the novel Pride and Prejudice? Answer with just the name."
echo
echo "===== EXPECTED: Paris / 391 / Jane Austen"
echo "===== Read them. All three right = serves and coherent."
echo "===== #1 right but #2/#3 wrong = PARTIAL DEGRADATION, do not ship."
echo "===== All wrong but fluent = codebook likely SLICED and the guard missed it."
