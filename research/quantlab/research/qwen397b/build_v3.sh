#!/bin/sh
# v3 candidate: the defect refund funds the promotions, so NO demotions.
#   promote {29,43,44,45,47,48} -> d4/K256   +1.125 GiB
#   convert 57/58/59 (affine 3-bit) -> d4/K2048  -1.125 GiB, and it IMPROVES
# Net: back at the shipped 100.971 GiB with six fewer damaged layers than v2.
# Promotions run FIRST: v2_sweep's preflight refuses a non-flat base, and the
# conversion makes it non-flat. demote_fit has no such constraint.
cd ~/v2sweep22 || exit 1
PY=/opt/homebrew/anaconda3/bin/python3
ARCH="/Volumes/Thunderbay HDD/vqlab-fits/qwen3.5-397b/demote_fit-d4"
SRC="/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16"

score() {
  for c in referee_corpus referee_corpus_code referee_corpus_literary; do
    printf "%s %s " "$2" "$c"
    $PY referee/score_streaming.py --model "$1" --corpus "referee/$c.txt" \
      --max-tokens 8192 2>/dev/null | grep -o '"ppl": [0-9.]*'
  done
}

echo "=== step 1: promote best-6 off the FLAT 2.2 base"
$PY v2_sweep.py --models-root ~/.exo/models \
  --base TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw --donor d4k256 \
  --combo 29,43,44,45,47,48 --tag V3P6 --workdir work \
  --state /tmp/v3-state.json --tsv /tmp/v3.tsv --build-only || exit 1

echo "=== step 2: convert 57/58/59 -> d4/K2048 (all archive hits)"
prev=work/397b-v2-V3P6
for L in 57 58 59; do
  nxt="work/v3-$L"
  $PY demote_fit.py --layer "$L" --k 2048 --dim 4 --base "$prev" --src "$SRC" \
    --save-fit "$ARCH" --stage-dir ~/v2sweep22/stage --out "$nxt" \
    || { echo "FAIL convert L$L"; exit 1; }
  rm -rf "$prev"
  prev="$nxt"
done
rm -rf work/397b-v3
mv "$prev" work/397b-v3
$PY -c "import json;print('SIZE', round(json.load(open('work/397b-v3/model.safetensors.index.json'))['metadata']['total_size']/2**30,3),'GiB  (shipped 100.971, iso100 100.964)')"
echo "=== smoke"
$PY -m mlx_lm generate --model work/397b-v3 \
  --prompt "Explain vector quantization in one sentence." --max-tokens 40 2>&1 | tail -8
score work/397b-v3 "V3"
echo V3 DONE
