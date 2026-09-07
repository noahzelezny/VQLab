#!/bin/sh
# PER-LAYER SHAPE SWEEP (overnight, 2026-09-06). For the top 16 layers by
# measured whole-layer effect, promote every non-trivial SUBSET of the three
# expert projections: 3 singles (0.0623 GiB) + 3 pairs (0.1245 GiB). The
# triple (0.1868) is already in sweep-2.2.tsv.
#
# WHY SUBSETS AND NOT A RULE: singles on L43/L45/L29/L47 showed every
# projection wins somewhere and additivity ranges 34-102%, so the shape is
# per-layer and the PAIRS are the unmeasured middle -- on L45 the singles sum
# to a third of the triple, meaning the projections interact and a pair may
# capture most of the layer for 2/3 of its bytes.
#
# TWO CORPORA PER ROW, not one. METHOD.md 13: single-window deltas carry ~35%
# relative sd, so a winner picked on one window may not be a winner. Each
# candidate is scored on the standard prose prefix AND on literary window 2
# (the hardest window, largest absolute deltas = best SNR). A subset only
# counts if it wins on BOTH.
#
# Promotions are deterministic splices of shipped 2.4 donor tensors -- no
# k-means anywhere, so a row is ~2 min build + 2 min scoring. Resumable:
# completed rows are skipped on restart.
cd ~/v2sweep22 || exit 1
PY=/opt/homebrew/anaconda3/bin/python3
TSV=shape-sweep.tsv
LAYERS="43 45 47 29 44 48 49 41 37 42 53 36 30 32 40 38"
SUBSETS="gate_proj up_proj down_proj gate_proj,up_proj gate_proj,down_proj up_proj,down_proj"
LITW=referee/windows/lit_w2.txt

[ -f "$TSV" ] || printf "candidate\tlayer\tsubset\tgib\tprose\tlit_w2\n" > "$TSV"
[ -f "$LITW" ] || { echo "missing $LITW"; exit 1; }

for L in $LAYERS; do
  for S in $SUBSETS; do
    short=$(echo "$S" | sed 's/_proj//g' | tr ',' '+')
    tag="L${L}_${short}"
    grep -q "^$tag	" "$TSV" 2>/dev/null && continue
    free=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
    [ "$free" -lt 12 ] && { echo "ABORT: only ${free}Gi free"; exit 1; }

    $PY v2_sweep.py --models-root ~/.exo/models \
      --base TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw --donor d4k256 \
      --combo "$L" --projections "$S" --tag "$tag" --workdir work \
      --state /tmp/shape-state.json --tsv /tmp/shape.tsv --build-only \
      >/dev/null 2>&1 || { echo "BUILD FAIL $tag"; continue; }

    dir="work/397b-v2-$tag"
    gib=$($PY -c "import json;print(round(json.load(open('$dir/model.safetensors.index.json'))['metadata']['total_size']/2**30,4))" 2>/dev/null)
    pp=$($PY referee/score_streaming.py --model "$dir" --corpus referee/referee_corpus.txt --max-tokens 8192 2>/dev/null | grep -o '"ppl": [0-9.]*' | cut -d' ' -f2)
    lp=$($PY referee/score_streaming.py --model "$dir" --corpus "$LITW" --max-tokens 8192 2>/dev/null | grep -o '"ppl": [0-9.]*' | cut -d' ' -f2)
    rm -rf "$dir"

    if [ -z "$pp" ] || [ -z "$lp" ]; then echo "SCORE FAIL $tag"; continue; fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$L" "$S" "$gib" "$pp" "$lp" >> "$TSV"
    echo "ROW $tag gib=$gib prose=$pp lit=$lp"
  done
done
echo SHAPE SWEEP DONE
