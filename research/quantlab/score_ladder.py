#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Score the escalating ladder and report a BREAKING-POINT CURVE per family.

The deliverable is not a total — it is the tier at which each model falls
off, so "which is better at something" has an answer with a location.
"""
import argparse
import json
import math
import pathlib
import re

HERE = pathlib.Path(__file__).parent
P = {p["id"]: p for p in json.load(open(HERE / "winrate" / "prompts_ladder.json"))}


def words(t):
    return re.findall(r"[A-Za-z0-9'’-]+", t)


def final_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if m:
        return m[-1].strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def norm(s):
    """Normalise for exact-match. The leading-article strip is NOT cosmetic:
    the state golds read "a brass coin" and models answer "brass coin", which
    the first version of this scorer marked WRONG — an instrument bug that
    would have reported a model failure that never happened (and unfairly
    penalised whichever model drops articles). Caught 2026-08-19 by reading
    the failures instead of trusting the count."""
    s = re.sub(r"\s+", " ", s.strip().strip(".!,:;\"'*_ ")).lower()
    return re.sub(r"^(a|an|the)\s+", "", s)


def exact_ok(pred, gold):
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    toks = re.findall(r"[a-z0-9]+", p)
    if g in toks and len(p) <= len(g) + 25:
        return True
    # Numeric golds: the LAST number on the ANSWER line is the commit.
    # The first version required the number to appear ALONE, which fails a
    # model that shows its arithmetic ("ANSWER: 23+45+12 = 80") despite the
    # final value being correct — punishing transparency. Trailing work is
    # fine; a different final number is not.
    if g.isdigit():
        nums = re.findall(r"\d+", p)
        return bool(nums) and nums[-1] == g
    return False


def check_constraints(text, c):
    t = text.strip()
    lines = [l for l in t.splitlines() if l.strip()]
    r = {}
    if "exact_lines" in c:
        r["lines"] = len(lines) == c["exact_lines"]
    if "max_words" in c:
        r["max_words"] = len(words(t)) <= c["max_words"]
    if "exact_words" in c:
        r["exact_words"] = len(words(t)) == c["exact_words"]
    if "start_with" in c:
        lead = re.sub(r'^[\s>*_"\'#\d.)-]+', "", t)
        r["start"] = lead.lower().startswith(c["start_with"].lower())
    if "must_include" in c:
        r["include"] = all(w.lower() in t.lower() for w in c["must_include"])
    if "forbid_words" in c:
        r["forbid"] = not any(re.search(rf"\b{re.escape(w)}\b", t, re.I)
                              for w in c["forbid_words"])
    if "end_with" in c:
        r["end"] = t.rstrip().endswith(c["end_with"])
    if "acrostic" in c:
        got = "".join(re.sub(r'^[\s>*_"\'#\d.)-]+', "", l)[:1].upper()
                      for l in lines)
        r["acrostic"] = got == c["acrostic"].upper()
    if "no_letter" in c:
        r["no_letter"] = c["no_letter"].lower() not in t.lower()
    if "all_sentences_start_capital" in c:
        r["caps"] = all(re.sub(r'^[\s>*_"\'#\d.)-]+', "", l)[:1].isupper()
                        for l in lines if l.strip())
    if "include_number" in c:
        r["digit"] = str(c["include_number"]) in t
    if "no_commas" in c:
        r["no_commas"] = "," not in t
    return r


def sustain_metrics(text, want):
    w = [x.lower() for x in words(text)]
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    sents = [norm(s) for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    eig = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    seen, longest, cur = set(), 0, 0
    for g in eig:
        if g in seen:
            cur += 1; longest = max(longest, cur)
        else:
            cur = 0; seen.add(g)
    return {"n": len(w), "reached": len(w) >= want * 0.9,
            "distinct3": len(set(tri)) / max(len(tri), 1),
            "rep_sent": 1 - len(set(sents)) / max(len(sents), 1),
            "rep8": longest}


def score(path):
    g = json.load(open(path))
    out = {"model": g["model"]}
    for it in g["gens"]:
        pr = P.get(it["id"])
        if not pr:
            continue
        d, tier, txt = pr["domain"], pr["tier"], it["text"]
        key = (d, tier)
        out.setdefault(key, [])
        if d in ("state", "chain", "needle", "needle_agg"):
            out[key].append(exact_ok(final_answer(txt), pr["answer"]))
        elif d == "constr":
            r = check_constraints(txt, pr["constraints"])
            out[key].append(all(r.values()))
            out.setdefault(("constr_parts", tier), []).extend(r.values())
        elif d == "sustain":
            out[key].append(sustain_metrics(txt, tier))
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n * 2)


def curve(s, label):
    print(f"\n{label}  ({s['model']})")
    KEYS = [k for k in s if isinstance(k, tuple)]
    fams = sorted({k[0] for k in KEYS})
    for fam in fams:
        if fam == "sustain":
            for tier in sorted(t for f, t in KEYS if f == fam):
                v = s[(fam, tier)]
                m = lambda k: sum(x[k] for x in v) / len(v)
                print(f"  {fam:11s} tier {tier:>5}: reached "
                      f"{sum(x['reached'] for x in v)}/{len(v)}  words "
                      f"{m('n'):.0f}  distinct3 {m('distinct3'):.3f}  "
                      f"rep-sent {m('rep_sent'):.3f}  rep8 {m('rep8'):.1f}")
            continue
        if fam == "constr_parts":
            continue
        row = []
        for tier in sorted(t for f, t in KEYS if f == fam):
            v = s[(fam, tier)]
            row.append(f"t{tier}:{sum(v)}/{len(v)}")
        print(f"  {fam:11s} " + "  ".join(row))
    if ("constr_parts", 12) in s:
        for tier in sorted(t for f, t in KEYS if f == "constr_parts"):
            v = s[("constr_parts", tier)]
            print(f"  constr-parts tier {tier:>2}: {sum(v)}/{len(v)} "
                  f"individual constraints")


ap = argparse.ArgumentParser()
ap.add_argument("--gens"); ap.add_argument("--a"); ap.add_argument("--b")
args = ap.parse_args()

if args.gens:
    curve(score(args.gens), "LADDER")
else:
    A, B = score(args.a), score(args.b)
    curve(A, "A"); curve(B, "B")
    print("\npaired McNemar by family (all tiers pooled)")
    for fam in ("state", "chain", "needle", "needle_agg", "constr"):
        b_ = c_ = n = 0
        for (f, t) in [k for k in A if isinstance(k, tuple)]:
            if f != fam or (f, t) not in B:
                continue
            for x, y in zip(A[(f, t)], B[(f, t)]):
                n += 1
                b_ += x and not y
                c_ += y and not x
        if n:
            print(f"  {fam:11s} n={n:3d}  A-only {b_:2d}, B-only {c_:2d}, "
                  f"p={mcnemar(b_, c_):.4f}")
