#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Decode a blind verdict against its withheld key; exact sign test.

Also reports the judge's RAW A/B split, because that is the check on the
instrument: a judge with a strong positional lean has to be read with the
lean's direction in mind (on 2026-08-18 Sonnet leaned to B while bf16 sat in
A for 34/60, so the bf16 win was if anything understated).
"""
import argparse, collections, json, math, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--verdict", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

P = pathlib.Path("winrate")
v = json.load(open(args.verdict))
k = json.load(open(P / f"blind_KEY_{args.tag}.json"))
key = k["key"]

def sign_p(w, l):
    n = w + l
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)

raw, dec, conf = collections.Counter(), collections.Counter(), collections.Counter()
for d in v["detail"]:
    raw[d["winner"]] += 1
    conf[d.get("confidence", "?")] += 1
    dec["tie" if d["winner"] == "tie" else key[str(d["pair_id"])][d["winner"]]] += 1
r, c = dec["ref"], dec["cand"]
p = sign_p(r, c)
print(f"=== {args.tag}  judge={v.get('judge','?')}  n={v.get('n', len(v['detail']))}")
print(f"  ref  : {k['ref']}")
print(f"  cand : {k['cand']}")
print(f"  raw positions : {dict(raw)}   (instrument check: strong lean?)")
print(f"  DECODED       : ref {r}  |  cand {c}  |  tie {dec['tie']}")
print(f"  confidence    : {dict(conf)}")
print(f"  sign test p   : {p:.4f}  -> {'SIGNIFICANT' if p <= .05 else 'no significant difference'}")
if r + c:
    print(f"  ref win-rate on decisive pairs: {r / (r + c):.1%}")
