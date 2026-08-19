#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Build winrate/prompts_ladder.json — an ESCALATING difficulty ladder whose
job is to find each model's BREAKING POINT, not to score them at ceiling.

WHY. E63's capacity probe returned 20/20, 10/10, 120/120 for BOTH models: a
ceiling result, which cannot rank. Noah: "push these guys until one starts
breaking." So every family here is emitted at four escalating tiers, and
the deliverable is the tier at which each model's accuracy falls off — a
curve, not a pass/fail. A family where both stay perfect at tier 4 gets
escalated again rather than reported as parity.

DESIGN RULE learned in E63: golds are GENERATED PROGRAMMATICALLY by a
simulator, never hand-written. One hand-written multihop item in E63 was
logically unsatisfiable and would have punished whichever model reasoned
correctly. Here the generator computes the answer from the same structure
it renders into prose, so the gold cannot disagree with the question.

Families (all exact-match scored except sustain, which is measured for
degeneration):
  state   — N sequential mutations of a world; ask final state. The
            classic small-model breaker: no shortcut, pure state carrying.
  chain   — N-1 adjacency links, shuffled, uniquely determining an order;
            ask position p. Depth of transitive chaining.
  constr  — 3/6/9/12 SIMULTANEOUS machine-checkable constraints.
  needle  — 8k/24k/48k token haystacks, plus multi-needle AGGREGATION
            (find 3 planted numbers, report their sum) which cannot be
            solved by a single lucky retrieval.
  sustain — 1500 and 3000 words, scored for looping.
"""
import json
import pathlib
import random

HERE = pathlib.Path(__file__).parent
rng = random.Random(20260819)
prompts = []

NAMES = ["Ada", "Bo", "Cyd", "Dov", "Eze", "Fen", "Gil", "Hana", "Ivo",
         "Jun", "Kit", "Lev", "Mira", "Nils", "Ola", "Pim"]
ITEMS = ["a red key", "a blue stone", "a brass coin", "a green feather",
         "a white pebble", "a silver ring", "a copper nail", "a black seed"]

# ------------------------------------------------------------------- state
# N boxes, a sequence of put/move/swap/empty ops. Simulator produces gold.
def make_state(n_ops, n_boxes, pid):
    boxes = {i: None for i in range(1, n_boxes + 1)}
    lines = []
    for _ in range(n_ops):
        op = rng.choice(["put", "move", "swap", "empty"])
        a, b = rng.sample(range(1, n_boxes + 1), 2)
        if op == "put":
            it = rng.choice(ITEMS)
            boxes[a] = it
            lines.append(f"Put {it} into box {a}.")
        elif op == "move":
            boxes[b] = boxes[a]
            boxes[a] = None
            lines.append(f"Move whatever is in box {a} into box {b}.")
        elif op == "swap":
            boxes[a], boxes[b] = boxes[b], boxes[a]
            lines.append(f"Swap the contents of box {a} and box {b}.")
        else:
            boxes[a] = None
            lines.append(f"Empty box {a}.")
    # ask about a box whose answer is unambiguous
    target = rng.randint(1, n_boxes)
    gold = boxes[target] if boxes[target] else "empty"
    body = (f"There are {n_boxes} numbered boxes, all empty to begin with. "
            f"Apply these operations in order:\n\n" + "\n".join(
                f"{i+1}. {l}" for i, l in enumerate(lines)))
    return {"id": pid, "domain": "state", "tier": n_ops, "passage": body,
            "task": f"After all {n_ops} operations, what is in box {target}? "
                    f"If it is empty, answer exactly: empty. End your reply "
                    f"with a final line of exactly this form:\n"
                    f"ANSWER: <contents>",
            "answer": gold}


pid = 3000
for tier, (n_ops, n_boxes) in enumerate([(10, 4), (25, 6), (50, 8), (100, 8)]):
    for _ in range(5):
        prompts.append(make_state(n_ops, n_boxes, pid)); pid += 1

# ------------------------------------------------------------------- chain
# A shuffled set of adjacency links uniquely fixes one order. Ask position p.
def make_chain(n, pid):
    people = rng.sample(NAMES, n)
    links = [f"{people[i]} stands immediately to the left of {people[i+1]}."
             for i in range(n - 1)]
    rng.shuffle(links)
    p = rng.randint(1, n)
    return {"id": pid, "domain": "chain", "tier": n,
            "passage": f"{n} people stand in a single row, left to right. "
                       f"You are told:\n\n" + "\n".join(links),
            "task": f"Who is standing in position {p}, counting from the "
                    f"left? End your reply with a final line of exactly this "
                    f"form:\nANSWER: <name>",
            "answer": people[p - 1]}


for n in (5, 8, 12, 16):
    for _ in range(5):
        prompts.append(make_chain(n, pid)); pid += 1

# ------------------------------------------------------------------ constr
POOL = [
    ("exact_lines", lambda: rng.choice([3, 4, 5])),
    ("max_words", lambda: rng.choice([70, 85, 100])),
    ("start_with", lambda: rng.choice(["Consider", "Begin", "Notice", "Every"])),
    ("must_include", lambda: rng.sample(["harbour", "lantern", "ledger",
                                         "tide", "rope", "salt"], 2)),
    ("forbid_words", lambda: rng.sample(["very", "really", "great", "nice"], 2)),
    ("end_with", lambda: "."),
    ("acrostic", lambda: rng.choice(["SALT", "TIDE", "ROPE"])),
    ("exact_words", lambda: rng.choice([40, 50, 60])),
    ("no_letter", lambda: rng.choice(["z", "q"])),
    ("all_sentences_start_capital", lambda: True),
    ("include_number", lambda: rng.randint(3, 9)),
    ("no_commas", lambda: True),
]
TOPICS = ["a lighthouse at dusk", "the morning fish market", "a ferry crossing",
          "an old shipping ledger", "a storm warning", "a harbour rope walk",
          "a tide table", "a salt marsh at dawn", "a chandlery shop",
          "a pilot boat", "a fog bell", "a quayside cafe", "a boatyard",
          "a weather station", "a net loft", "a customs house",
          "a coastal footpath", "a bell buoy", "a slipway", "a chart room"]


def render(cons):
    r = []
    if "exact_lines" in cons:
        r.append(f"Write exactly {cons['exact_lines']} lines, one sentence "
                 f"per line, no bullets or numbering.")
    if "max_words" in cons:
        r.append(f"Use at most {cons['max_words']} words in total.")
    if "exact_words" in cons:
        r.append(f"Use exactly {cons['exact_words']} words in total.")
    if "start_with" in cons:
        r.append(f"The first line must begin with \"{cons['start_with']}\".")
    if "must_include" in cons:
        r.append(f"Include these words: {', '.join(cons['must_include'])}.")
    if "forbid_words" in cons:
        r.append(f"Do not use: {', '.join(cons['forbid_words'])}.")
    if "end_with" in cons:
        r.append("The final line must end with a full stop.")
    if "acrostic" in cons:
        r.append(f"The first letters of the lines, read downward, must spell "
                 f"{cons['acrostic']}.")
    if "no_letter" in cons:
        r.append(f"Do not use the letter '{cons['no_letter']}' anywhere.")
    if "all_sentences_start_capital" in cons:
        r.append("Every line must begin with a capital letter.")
    if "include_number" in cons:
        r.append(f"The digit {cons['include_number']} must appear.")
    if "no_commas" in cons:
        r.append("Do not use any commas.")
    return " ".join(r)


ti = 0
for k in (3, 6, 9, 12):
    for _ in range(5):
        keys = ["exact_lines", "max_words", "start_with"][:min(3, k)]
        extra = [x for x in POOL if x[0] not in keys]
        rng.shuffle(extra)
        for name, _f in extra[:max(0, k - 3)]:
            keys.append(name)
        if "acrostic" in keys and "exact_lines" not in keys:
            keys.append("exact_lines")
        cons = {}
        for name in keys:
            cons[name] = dict(POOL)[name]()
        if "acrostic" in cons:
            cons["exact_lines"] = len(cons["acrostic"])
        if "exact_words" in cons and "max_words" in cons:
            del cons["max_words"]
        prompts.append({"id": pid, "domain": "constr", "tier": k,
                        "passage": f"Write a short description of "
                                   f"{TOPICS[ti % len(TOPICS)]}.",
                        "task": render(cons), "constraints": cons})
        pid += 1; ti += 1

# ------------------------------------------------------------------ needle
corpus = (HERE / "referee" / "referee_corpus_literary.txt").read_text()
paras = [p for p in corpus.split("\n\n") if len(p) > 400]
CH_PER = {8: 32000, 24: 96000, 48: 192000}
k = 0
for ktok in (8, 24, 48):
    for j in range(3):
        want = CH_PER[ktok]
        body, tot = [], 0
        while tot < want and k < len(paras):
            body.append(paras[k]); tot += len(paras[k]); k += 1
        num = rng.randint(101, 989)
        fact = (f"The harbour ledger for that season recorded exactly {num} "
                f"brass lanterns held in the east vault.")
        pos = max(1, int(len(body) * [0.15, 0.5, 0.85][j]))
        body = body[:pos] + [fact] + body[pos:]
        prompts.append({"id": pid, "domain": "needle", "tier": ktok,
                        "passage": "\n\n".join(body),
                        "task": "How many brass lanterns were recorded in the "
                                "east vault? End with a line of exactly this "
                                "form:\nANSWER: <number>",
                        "answer": str(num)})
        pid += 1
    # multi-needle aggregation at the same size — no single lucky retrieval
    for j in range(2):
        want = CH_PER[ktok]
        body, tot = [], 0
        while tot < want and k < len(paras):
            body.append(paras[k]); tot += len(paras[k]); k += 1
        nums = [rng.randint(11, 89) for _ in range(3)]
        facts = [f"The {w} ledger recorded {n} brass lanterns."
                 for w, n in zip(["spring", "summer", "autumn"], nums)]
        for d, f in zip([0.2, 0.55, 0.9], facts):
            body.insert(max(1, int(len(body) * d)), f)
        prompts.append({"id": pid, "domain": "needle_agg", "tier": ktok,
                        "passage": "\n\n".join(body),
                        "task": "Three ledgers (spring, summer, autumn) each "
                                "recorded a number of brass lanterns. What is "
                                "the SUM of those three numbers? End with a "
                                "line of exactly this form:\nANSWER: <number>",
                        "answer": str(sum(nums))})
        pid += 1

# ----------------------------------------------------------------- sustain
lit = json.load(open(HERE / "winrate" / "prompts.json"))
for t_i, words_n in enumerate((1500, 3000)):
    for j in range(4):
        prompts.append({"id": pid, "domain": "sustain", "tier": words_n,
                        "passage": lit[(t_i * 4 + j) * 5]["passage"],
                        "task": f"Continue this passage for at least "
                                f"{words_n} words, matching its voice, "
                                f"register and period. Do not summarize, do "
                                f"not stop early, do not repeat yourself or "
                                f"restate earlier sentences. Output only the "
                                f"continuation."})
        pid += 1

out = HERE / "winrate" / "prompts_ladder.json"
out.write_text(json.dumps(prompts, indent=1))
by = {}
for p in prompts:
    by.setdefault(p["domain"], []).append(p["tier"])
print(f"{len(prompts)} prompts -> {out}")
for d, t in by.items():
    print(f"  {d:11s} n={len(t):3d}  tiers {sorted(set(t))}")
mx = max(len(p["passage"]) for p in prompts)
print(f"  longest prompt {mx} chars (~{mx//4} tokens)")
