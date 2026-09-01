#!/bin/sh
# E129 — score the M4-fitted vintage arm. Matches E121's invocation exactly
# (run_e121_oldfitter.sh:71-72): same referee, same two corpora, so the arms
# are comparable. K=256 is byte-aligned so pack is a no-op and there is no
# "-packed" dir; scoring the artifact directory IS scoring the final bytes.
# NOT SMOKED: 110.768 GiB against a 96 GiB box violates III.11a. The streaming
# referee scores larger-than-RAM by design; generation cannot. This is an
# experiment arm, not a release candidate, so III.11 does not gate it.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
S="$E/rotlab--397B-flatk256-aug15stack-b"
L=logs_live_e136b_aug15.log
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a $L; }
say "=== E136b (draw 2) scoring: $(basename "$S")"
$V - <<PY 2>&1 | tee -a $L
import os
d="$S"
print("MEASURED SIZE %s: %.3f GiB"%(os.path.basename(d),
  sum(os.path.getsize(os.path.join(d,f)) for f in os.listdir(d) if f.endswith(".safetensors"))/2**30))
PY
say "--- wikitext"
$V referee/score_streaming.py --model "$S" --corpus referee/referee_corpus.txt 2>&1 | tee -a $L | tail -1
say "--- code"
$V referee/score_streaming.py --model "$S" --corpus referee/referee_corpus_code.txt 2>&1 | tee -a $L | tail -1
say "=== E136b SCORED"
