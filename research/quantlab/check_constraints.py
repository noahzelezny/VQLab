#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Machine-score the instruction-following gens (ids 1000+) — pass/fail per
constraint, no judge model. This is the deterministic lens in the E56
gemma-small protocol: at n prompts a judge adds variance, a checker adds
none, and pass/fail pairs directly into McNemar alongside litbench.

    ./check_constraints.py --gens winrate/gens_domains_<model>.json
    ./check_constraints.py --a <gens> --b <gens>      # paired McNemar
"""
import argparse
import json
import math
import pathlib
import re

HERE = pathlib.Path(__file__).parent
PROMPTS = {p["id"]: p for p in
           json.load(open(HERE / "winrate" / "prompts_domains.json"))
           if "constraints" in p}


def words(t):
    return re.findall(r"[A-Za-z0-9'’-]+", t)


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


def check(text, cons):
    t = text.strip()
    if "max_words" in cons and len(words(t)) > cons["max_words"]:
        return False, f"{len(words(t))} words > {cons['max_words']}"
    if "max_sentences" in cons and len(sentences(t)) > cons["max_sentences"]:
        return False, f"{len(sentences(t))} sentences > {cons['max_sentences']}"
    if "exact_lines" in cons:
        lines = [l for l in t.splitlines() if l.strip()]
        if len(lines) != cons["exact_lines"]:
            return False, f"{len(lines)} lines != {cons['exact_lines']}"
    if "start_with" in cons:
        # tolerate markdown bold/quote wrapping, nothing else
        lead = re.sub(r'^[\s>*_"\'#-]+', "", t)
        if not lead.startswith(cons["start_with"]):
            return False, f"does not start with {cons['start_with']!r}"
    if "forbid_char" in cons and cons["forbid_char"] in t.lower():
        return False, f"contains {cons['forbid_char']!r}"
    return True, ""


def score(path):
    g = json.load(open(path))
    res = {}
    for it in g["gens"]:
        if it["id"] not in PROMPTS:
            continue
        ok, why = check(it["text"], PROMPTS[it["id"]]["constraints"])
        res[it["id"]] = (ok, why)
    return g["model"], res


def mcnemar(b, c):
    """Exact two-sided binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


ap = argparse.ArgumentParser()
ap.add_argument("--gens")
ap.add_argument("--a")
ap.add_argument("--b")
args = ap.parse_args()

if args.gens:
    name, res = score(args.gens)
    npass = sum(ok for ok, _ in res.values())
    print(f"{name}: {npass}/{len(res)} constraints passed")
    for pid, (ok, why) in sorted(res.items()):
        if not ok:
            print(f"    FAIL {pid}: {why}")
else:
    na, ra = score(args.a)
    nb, rb = score(args.b)
    ids = sorted(set(ra) & set(rb))
    b_ = sum(1 for i in ids if ra[i][0] and not rb[i][0])
    c_ = sum(1 for i in ids if rb[i][0] and not ra[i][0])
    both = sum(1 for i in ids if ra[i][0] and rb[i][0])
    print(f"A={na}: {sum(ok for ok,_ in ra.values())}/{len(ids)}   "
          f"B={nb}: {sum(ok for ok,_ in rb.values())}/{len(ids)}")
    print(f"paired: both-pass {both}, A-only {b_}, B-only {c_}, "
          f"McNemar exact p = {mcnemar(b_, c_):.4f}")
