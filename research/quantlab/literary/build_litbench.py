#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Merge, validate, and deterministically shuffle the literary item shards.

WHY A BUILD STEP. The shards are hand-authored, one file per pair of
categories, every item written with the correct answer first — that is the
only way to author distractors honestly, because you write the right answer
and then attack it. But an artifact where `label == 0` for all 104 items is
a trap waiting for the first person who scores it with anything other than
independent-continuation loglikelihood. So the shipped file is shuffled,
once, with a fixed seed, and the shuffle is reproducible from the shards.

WHY POSITION DOES NOT (CURRENTLY) MATTER. lm-eval's `multiple_choice`
output type scores each choice as its own (context, continuation) pair —
`score_tasks_streaming.py:200` `_loglikelihood_pairs`. The choices are never
presented to the model together, so ordering cannot leak the answer. The
shuffle is hygiene for any future use (a generative or judged variant),
not a correction to the current scoring path.

VALIDATION IS THE POINT. A literary benchmark fails silently: if the right
answer is systematically the longest or most fluent option, the bench
measures surface statistics and reports it as literary understanding. The
checks below are the guard, and `--report` prints the evidence.

    ./build_litbench.py                 # build + validate
    ./build_litbench.py --report        # + per-category length/lexical stats
"""
import argparse
import collections
import json
import pathlib
import random
import statistics

HERE = pathlib.Path(__file__).parent
SHARDS = sorted(HERE.glob("items_*.jsonl"))
OUT = HERE / "litbench.jsonl"
SEED = 20260817          # fixed: the shuffle must be reproducible from shards
N_CHOICES = 4


def load_shards():
    items, seen = [], set()
    for shard in SHARDS:
        for lineno, line in enumerate(shard.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{shard.name}:{lineno}: bad JSON — {e}")
            it["_src"] = f"{shard.name}:{lineno}"
            if it["id"] in seen:
                raise SystemExit(f"{it['_src']}: duplicate id {it['id']}")
            seen.add(it["id"])
            items.append(it)
    return items


def validate(items):
    """Hard structural checks. Any failure aborts the build."""
    errs = []
    for it in items:
        w = f"{it['_src']} [{it['id']}]"
        for field in ("id", "category", "passage", "question", "choices", "label"):
            if field not in it:
                errs.append(f"{w}: missing field {field!r}")
        ch = it.get("choices", [])
        if len(ch) != N_CHOICES:
            errs.append(f"{w}: {len(ch)} choices, expected {N_CHOICES}")
        for i, c in enumerate(ch):
            if not isinstance(c, str) or not c.strip():
                errs.append(f"{w}: choice {i} is empty or not a string")
        if len(set(ch)) != len(ch):
            errs.append(f"{w}: duplicate choice text")
        if not isinstance(it.get("label"), int) or not 0 <= it["label"] < len(ch):
            errs.append(f"{w}: label {it.get('label')!r} out of range")
        # A choice punctuated differently from its siblings is a giveaway:
        # the model can pick the odd one out on form alone. Compare whether
        # each choice ENDS in punctuation, not which character it ends on
        # (these are unpunctuated fragments, so the last letter varies freely).
        ends = {c.rstrip()[-1] in ".!?\"'" for c in ch if c.strip()}
        if len(ends) > 1:
            errs.append(f"{w}: some choices end in punctuation and some do not")
    if errs:
        raise SystemExit("VALIDATION FAILED:\n  " + "\n  ".join(errs))


def length_bias(items):
    """Is the correct answer systematically longer/shorter than distractors?

    This is the failure mode that makes a literary bench meaningless. We
    report the mean rank of the answer's character length among its four
    choices; 2.5 is unbiased, 4.0 means 'always the longest'.
    """
    ranks, deltas = [], []
    for it in items:
        lens = [len(c) for c in it["choices"]]
        gold = lens[it["label"]]
        ranks.append(sorted(lens).index(gold) + 1)
        others = [l for i, l in enumerate(lens) if i != it["label"]]
        deltas.append(gold - statistics.mean(others))
    return statistics.mean(ranks), statistics.mean(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    items = load_shards()
    validate(items)

    rng = random.Random(SEED)
    out = []
    for it in items:
        gold = it["choices"][it["label"]]
        ch = list(it["choices"])
        rng.shuffle(ch)
        # NO leading space on choices. lm-eval's multiple_choice path already
        # emits the continuation as " {choice}" when it builds the request
        # (verified: prefixing here produced a DOUBLE space at the seam, which
        # re-tokenizes differently and would silently shift every score).
        # `doc_to_text` therefore ends on "Answer:" with no trailing space.
        out.append({
            "id": it["id"],
            "category": it["category"],
            "passage": it["passage"],
            "question": it["question"],
            "choices": ch,
            "label": ch.index(gold),
        })

    OUT.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))

    cats = collections.Counter(o["category"] for o in out)
    labs = collections.Counter(o["label"] for o in out)
    mean_rank, mean_delta = length_bias(out)

    print(f"wrote {OUT}  ({len(out)} items, {len(SHARDS)} shards)")
    print(f"categories: {dict(sorted(cats.items()))}")
    print(f"label positions: {dict(sorted(labs.items()))}  (want ~uniform)")
    print(f"answer length-rank: {mean_rank:.2f} / 4  (2.50 = unbiased)")
    print(f"answer vs distractor mean chars: {mean_delta:+.1f}")

    if args.report:
        print("\nper-category answer length-rank:")
        for c in sorted(cats):
            sub = [o for o in out if o["category"] == c]
            r, d = length_bias(sub)
            print(f"  {c:22s} n={len(sub):3d}  rank {r:.2f}  delta {d:+6.1f}")

    # POWER. Units are ACCURACY POINTS (pp) — the gap in percent-correct that
    # a paired McNemar test can distinguish from noise. Not perplexity.
    #
    # Resolving power is 1.96*sqrt(d/n), where d is the DISCORDANCE RATE (how
    # often the two models disagree per item). d is not a constant, and using
    # one value for both regimes is the mistake this note used to make:
    #
    #   comparing two DIFFERENT models   -> they disagree a lot, d ~ 30%
    #   comparing two QUANTS of one model -> they agree on nearly everything.
    #     Measured on this project's own 397B task-bench runs: 44/1000 and
    #     55/1000 discordant, i.e. d = 4.4-5.5%. See analyze_task_bench.py
    #     output, the "disc W/L" column.
    #
    # The quant case is the SENSITIVE one, not the hard one — near-identical
    # models produce few discordant pairs, and each one is informative.
    n = len(out)
    print(f"\nPOWER NOTE (accuracy points, not perplexity): n={n}")
    for d, lbl in ((0.30, "vs a different model  (d~30%)"),
                   (0.05, "vs a quant of the same (d~5%)")):
        pp = 1.96 * (d / n) ** 0.5 * 100
        print(f"  {lbl}: resolves gaps down to ~{pp:.1f}pp")
    need = 0.05 * (1.96 / 0.02) ** 2
    print(f"  to resolve a 2pp quant delta you would need ~{need:.0f} items.")


if __name__ == "__main__":
    main()
