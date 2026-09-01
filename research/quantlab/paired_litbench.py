#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Paired McNemar on two litbench runs that stored per_item (E56).

Two independent accuracies at n=104 carry ~±3.7pt SE each; pairing on the
same items removes the item-difficulty variance, which is most of it. The
e4b-8bit "beats" gemma-small verdict rests on 4.8pts — this is the test
with any chance of resolving that at this n.

    ./paired_litbench.py results_literary/gencyc_A.json gencyc_B.json
"""
import json
import math
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
pa = {i["id"]: i["correct"] for i in a["per_item"]}
pb = {i["id"]: i["correct"] for i in b["per_item"]}
ids = sorted(set(pa) & set(pb))
b_ = sum(1 for i in ids if pa[i] and not pb[i])   # A right, B wrong
c_ = sum(1 for i in ids if pb[i] and not pa[i])
n = b_ + c_
p = min(1.0, sum(math.comb(n, i) for i in range(min(b_, c_) + 1)) / 2**n * 2) if n else 1.0
print(f"A={a['model']}  acc {a['accuracy']:.4f}")
print(f"B={b['model']}  acc {b['accuracy']:.4f}")
print(f"paired n={len(ids)}: A-only-right {b_}, B-only-right {c_}, "
      f"discordant {n}, McNemar exact p={p:.4f}")
