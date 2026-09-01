#!/bin/sh
# Single follow-on: referee the SHIPPED 2.2 and 3.1 on the same harness the
# ladder used, so every comparison row is same-instrument. Waits for the
# ladder; one waiter only (stampede rule).
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
while pgrep -f "run_ladder_397b.sh" >/dev/null; do sleep 120; done
for n in VQ-2.2bpw VQ-3.1bpw; do
  for c in referee_corpus.txt referee_corpus_code.txt; do
    $V referee/score_streaming.py --model "$E/TheDrainFlorist--Qwen3.5-397B-A17B-$n" --corpus referee/$c 2>&1 | tail -1
  done
done
echo "########## REFS DONE"
