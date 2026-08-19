#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Score the capacity probe set — three families machine-checked, one
mechanically measured for degeneration. No judge model anywhere.

    ./score_capacity.py --gens winrate/gens_capacity_<model>.json
    ./score_capacity.py --a <gens> --b <gens>        # paired, per family

WHY MECHANICAL. Every judged instrument we own returned parity between a
2.25bpw 26B and an 8-bit 4B, and a judge adds variance we cannot afford at
n=20 per family. Here: multihop and needle are exact-match against golds
fixed before generation; constraint is per-constraint pass/fail (partial
credit, so a model that satisfies 4 of 6 is distinguishable from one that
satisfies 2); sustain is scored for LOOPING, which is what a model out of
capacity actually does — distinct-trigram ratio, repeated-sentence rate,
longest repeated span, and whether it reached length at all.
"""
import argparse
import json
import math
import pathlib
import re

HERE = pathlib.Path(__file__).parent
PROMPTS = {p["id"]: p for p in
           json.load(open(HERE / "winrate" / "prompts_capacity.json"))}


def words(t):
    return re.findall(r"[A-Za-z0-9'’-]+", t)


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def final_answer(text):
    """Last ANSWER: line wins — models restate; the last one is the commit."""
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if not m:
        # fall back to the last non-empty line, so a model that answered
        # correctly but ignored the format is not scored as wrong for format.
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        return lines[-1] if lines else ""
    return m[-1].strip()


def norm(s):
    s = s.strip().strip(".!,:;\"'*_ ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def exact_ok(pred, gold):
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    # a bare gold appearing as its own token inside a short answer counts
    return g in re.findall(r"[A-Za-z0-9:]+", p) and len(p) <= len(g) + 25


def score_constraints(text, cons):
    """Per-constraint pass/fail -> partial credit."""
    t = text.strip()
    res = {}
    lines = [l for l in t.splitlines() if l.strip()]
    res["lines"] = len(lines) == cons["exact_lines"]
    res["words"] = len(words(t)) <= cons["max_words"]
    lead = re.sub(r'^[\s>*_"\'#\d.)-]+', "", t)
    res["start"] = lead.lower().startswith(cons["start_with"].lower())
    res["include"] = all(w.lower() in t.lower() for w in cons["must_include"])
    res["forbid"] = not any(w.lower() in t.lower() for w in cons["forbid_words"])
    res["end"] = t.rstrip().endswith(cons["end_with"])
    return res


def sustain_metrics(text):
    w = [x.lower() for x in words(text)]
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    distinct3 = len(set(tri)) / max(len(tri), 1)
    sents = [norm(s) for s in sentences(text)]
    rep_sent = 1 - (len(set(sents)) / max(len(sents), 1))
    # longest run of repeated 8-grams (a looping model spikes this)
    eig = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    seen, longest, cur = set(), 0, 0
    for g in eig:
        if g in seen:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
            seen.add(g)
    return {"n_words": len(w), "distinct3": distinct3,
            "repeat_sent": rep_sent, "longest_rep8": longest,
            "reached_800": len(w) >= 800}


def score(path):
    g = json.load(open(path))
    out = {"model": g["model"], "multihop": {}, "constraint": {},
           "needle": {}, "sustain": {}}
    for it in g["gens"]:
        pr = PROMPTS.get(it["id"])
        if not pr:
            continue
        d, txt = pr["domain"], it["text"]
        if d in ("multihop", "needle"):
            out[d][it["id"]] = exact_ok(final_answer(txt), pr["answer"])
        elif d == "constraint":
            out[d][it["id"]] = score_constraints(txt, pr["constraints"])
        elif d == "sustain":
            out[d][it["id"]] = sustain_metrics(txt)
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def report(s):
    mh = s["multihop"]
    nd = s["needle"]
    print(f"{s['model']}")
    if mh:
        print(f"  multihop  {sum(mh.values())}/{len(mh)}")
    if nd:
        byd = {}
        for pid, ok in nd.items():
            byd[PROMPTS[pid]["depth"]] = byd.get(PROMPTS[pid]["depth"], []) + [ok]
        print(f"  needle    {sum(nd.values())}/{len(nd)}  " +
              " ".join(f"d{int(k*100)}:{sum(v)}/{len(v)}"
                       for k, v in sorted(byd.items())))
    cs = s["constraint"]
    if cs:
        tot = sum(sum(v.values()) for v in cs.values())
        n = sum(len(v) for v in cs.values())
        allpass = sum(all(v.values()) for v in cs.values())
        per = {}
        for v in cs.values():
            for k, ok in v.items():
                per[k] = per.get(k, 0) + ok
        print(f"  constraint {tot}/{n} individual, {allpass}/{len(cs)} perfect"
              f"   [{' '.join(f'{k}:{v}' for k, v in per.items())}]")
    su = s["sustain"]
    if su:
        m = lambda k: sum(v[k] for v in su.values()) / len(su)
        print(f"  sustain   mean words {m('n_words'):.0f}, reached800 "
              f"{sum(v['reached_800'] for v in su.values())}/{len(su)}, "
              f"distinct3 {m('distinct3'):.3f}, repeat-sent "
              f"{m('repeat_sent'):.3f}, longest-rep8 {m('longest_rep8'):.1f}")


ap = argparse.ArgumentParser()
ap.add_argument("--gens")
ap.add_argument("--a")
ap.add_argument("--b")
args = ap.parse_args()

if args.gens:
    report(score(args.gens))
else:
    A, B = score(args.a), score(args.b)
    report(A)
    report(B)
    print("\npaired tests")
    for fam in ("multihop", "needle"):
        ids = sorted(set(A[fam]) & set(B[fam]))
        b_ = sum(1 for i in ids if A[fam][i] and not B[fam][i])
        c_ = sum(1 for i in ids if B[fam][i] and not A[fam][i])
        print(f"  {fam:10s} n={len(ids)}  A-only {b_}, B-only {c_}, "
              f"McNemar p={mcnemar(b_, c_):.4f}")
    ids = sorted(set(A["constraint"]) & set(B["constraint"]))
    b_ = c_ = 0
    for i in ids:
        for k in A["constraint"][i]:
            if A["constraint"][i][k] and not B["constraint"][i][k]:
                b_ += 1
            elif B["constraint"][i][k] and not A["constraint"][i][k]:
                c_ += 1
    print(f"  constraint per-constraint discordant: A-only {b_}, B-only {c_},"
          f" McNemar p={mcnemar(b_, c_):.4f}")
