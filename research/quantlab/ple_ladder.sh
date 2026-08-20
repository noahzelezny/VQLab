#!/bin/sh
# E66b: geometry ladder on the PLE table. 154s per rung, and PLE is 35.5% of
# the model — the cheapest high-leverage sweep available.
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
SRC="/Volumes/Thunderbay SSD/Mlx_Models/hub/models--mlx-community--gemma-4-e4b-it-bf16/snapshots/eec12d0899edea9b738ab1009af9159cdfd70d71"
for g in "2 256" "2 1024" "4 2048" "4 512" "4 256"; do
  d=$(echo $g|cut -d' ' -f1); k=$(echo $g|cut -d' ' -f2)
  echo "########## PLE d${d}-K${k}"
  ./venv/bin/python fit_e4b_ple.py --src "$SRC" \
     --out "/Volumes/Thunderbay SSD/Exo Models/e4b-vq-PLE-d${d}k${k}" \
     --dim $d --k $k 2>&1 | tee -a logs_live_ple_ladder.log | tail -1
done
echo "########## PLE LADDER DONE"
