#!/bin/sh
# Re-run ONLY the repaired constraint family, both models, after the ladder
# run releases the GPU. Merges back into the existing gens files so the
# other families' generations are not wasted.
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
while pgrep -f "run_ladder.sh" >/dev/null; do sleep 60; done
$V - <<'PY'
import json, pathlib
P=json.load(open('winrate/prompts_ladder.json'))
c=[p for p in P if p['domain']=='constr']
json.dump(c, open('winrate/prompts_constr_fixed.json','w'), indent=1)
print(f"{len(c)} repaired constraint prompts")
PY
for pair in "small:$E/gemma26b-rungs/vq-K256-d4:" \
            "e4b8:$E/mlx-community--gemma-4-e4b-it-8bit:--allow-unmatched"; do
  tag=$(echo "$pair"|cut -d: -f1); mdl=$(echo "$pair"|cut -d: -f2); flag=$(echo "$pair"|cut -d: -f3)
  $V winrate_bench.py generate --model "$mdl" --prompts winrate/prompts_constr_fixed.json \
     --max-tokens 900 $flag --out winrate/gens_constr_$tag.json 2>&1 | tee -a logs_live_$(basename rerun_constr.sh .sh).log | tail -1
  $V - "$tag" <<'PY'
import json,sys
tag=sys.argv[1]
base=json.load(open(f'winrate/gens_ladder_{tag}.json'))
new={g['id']:g for g in json.load(open(f'winrate/gens_constr_{tag}.json'))['gens']}
base['gens']=[new.get(g['id'],g) for g in base['gens']]
base['note']='constraint family regenerated after 11 unsatisfiable items were repaired'
json.dump(base,open(f'winrate/gens_ladder_{tag}.json','w'),indent=1)
print(f"merged {len(new)} repaired constraint gens into gens_ladder_{tag}.json")
PY
done
echo "===== PAIRED LADDER RESULT (constraints repaired) ====="
$V score_ladder.py --a winrate/gens_ladder_small.json --b winrate/gens_ladder_e4b8.json
