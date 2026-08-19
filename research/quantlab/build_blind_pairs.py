#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Build anonymized A/B pair files for blind judging, key withheld.

WHY. A judging session that can see `gens_bf16.json` in the filename knows
which side is the reference, so blindness has to be STRUCTURAL, not a polite
request. This randomizes which side each model lands on, per pair, and writes
the key to a SEPARATE file the judge is never pointed at.

Decode afterwards with score_blind_verdict.py, which maps A/B back to model
names and runs an exact two-sided sign test on decisive pairs.

    ./build_blind_pairs.py --ref winrate/gens_bf16.json \
        --cand winrate/gens_X.json --tag X
"""
import argparse, json, pathlib, random

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="reference generations (bf16)")
ap.add_argument("--cand", required=True, help="candidate generations")
ap.add_argument("--tag", required=True)
ap.add_argument("--prompts", default="winrate/prompts.json")
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()

P = pathlib.Path("winrate")
prompts = {p["id"]: p for p in json.load(open(args.prompts))}
ref = {g["id"]: g["text"] for g in json.load(open(args.ref))["gens"]}
cand = {g["id"]: g["text"] for g in json.load(open(args.cand))["gens"]}
rng = random.Random(args.seed)

blind, key = [], {}
for i in sorted(set(ref) & set(cand)):
    flip = rng.random() < 0.5
    A, B = (ref[i], cand[i]) if flip else (cand[i], ref[i])
    blind.append({"pair_id": i, "passage": prompts[i]["passage"], "A": A, "B": B})
    key[str(i)] = {"A": "ref" if flip else "cand", "B": "cand" if flip else "ref"}

(P / f"blind_pairs_{args.tag}.json").write_text(json.dumps(blind, indent=1))
(P / f"blind_KEY_{args.tag}.json").write_text(json.dumps(
    {"ref": args.ref, "cand": args.cand, "key": key}, indent=1))
n_a = sum(1 for v in key.values() if v["A"] == "ref")
print(f"{args.tag}: {len(blind)} pairs, ref in A for {n_a}/{len(key)} "
      f"(key withheld in blind_KEY_{args.tag}.json)")
