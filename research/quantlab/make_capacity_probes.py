#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Build winrate/prompts_capacity.json — the probe set designed to SEPARATE.

WHY THIS EXISTS. Every instrument we own returned parity between a 2.25bpw
26B and an 8-bit 4B: litbench McNemar p=0.33, teacher-vs-quant p=0.15,
constraint pass-rate 19/20 vs 19/20 p=1.0, and my own read of 4 prose pairs
scored ~2-1-1. Noah's objection is the right one: that cannot be the whole
truth about a 6x parameter gap. The resolution is that NONE of those tasks
stress capacity. "Why does the moon follow the car" needs one good analogy,
not 26B parameters.

So this set is built the other way round: every item is chosen because a
small model should run OUT OF ROOM on it. Four families, and THREE of the
four are machine-scored so the verdict does not rest on a judge:

  A. multihop  (20) — constructed-world chains, exact answer, no world
     knowledge involved so it is pure reasoning depth. An early wrong turn
     poisons the rest, which is exactly the failure a small model makes.
  B. constraint(20) — SIX simultaneous machine-checkable constraints per
     item, vs one in the easy set. Small models satisfy 1-2 and drop the
     rest; scoring is per-constraint so we see partial credit.
  C. needle   (10) — a distinctive fact planted at a known depth in a long
     literary haystack; exact-match retrieval. Tests long-context tracking.
  D. sustain  (10) — 800-word continuations, scored MECHANICALLY for
     degeneration (distinct-n, sentence repetition, longest repeat) rather
     than by taste. Small models loop; that is measurable without a judge.

IDs are namespaced 2000+ so they cannot collide with the literary (0-59) or
domain (1000+) sets.
"""
import json
import pathlib
import random
import re

HERE = pathlib.Path(__file__).parent
prompts = []

# ---------------------------------------------------------------- A. multihop
# Constructed worlds: every fact needed is IN the prompt, so this measures
# reasoning depth, not recall. Answers are exact tokens.
MULTIHOP = [
    ("Five crates sit in a row on a dock: the anchor is left of the rope, "
     "the rope is left of the net, the lamp is immediately right of the net, "
     "and the barrel is at the far right. Nothing else moves. Which crate is "
     "exactly in the middle of the row?", "net"),
    ("Ana is twice Ben's age. Ben is 4 years younger than Cleo. Cleo will be "
     "20 in three years. How old is Ana now?", "26"),
    ("A train leaves at 09:40 and the trip takes 2 hours 50 minutes. The "
     "traveller then waits 35 minutes and takes a second trip of 1 hour 15 "
     "minutes. At what time does the traveller arrive? Use 24-hour time.",
     "14:20"),
    ("Every gronk is a fleeb. Some fleebs are murls. All murls are quiet. "
     "Bix is a gronk and is not quiet. Is Bix a murl? Answer yes or no.",
     "no"),
    ("In a shop, a widget costs 3 coins more than a gadget. Two widgets and "
     "three gadgets cost 41 coins. How many coins does one widget cost?",
     "10"),
    # NOTE: the first draft of this item was UNSATISFIABLE (Dara<Eli<Faye<Gus
    # forced Gus last while the item also asserted "Gus is not last"). Caught
    # by hand-verifying every gold answer before the run. Verify golds.
    ("Four runners finish a race. Dara finishes ahead of Eli. Gus finishes "
     "behind Dara but ahead of Faye. Eli finishes behind Faye. Who finished "
     "last?", "Eli"),
    ("A tank holds 96 litres. It drains at 4 litres per minute for 9 minutes, "
     "then is refilled at 6 litres per minute for 5 minutes. How many litres "
     "are in the tank then?", "90"),
    ("Books are shelved alphabetically by author. Quinn is shelved before "
     "Ruiz. Patel is shelved before Quinn. Osei is shelved immediately "
     "before Patel. Which author is second from the start?", "Patel"),
    ("A clock loses 3 minutes every hour. It is set correctly at 08:00. What "
     "does it read when the true time is 13:00? Use 24-hour time.", "12:45"),
    ("If it rains, the match is indoors. If the match is indoors, Tam "
     "referees. Tam did not referee. Did it rain? Answer yes or no.", "no"),
    ("A bag has 3 red, 5 blue and 4 green tokens. All green tokens are "
     "removed, then half the blue ones are removed (rounding down). How many "
     "tokens remain in the bag?", "6"),
    ("Mira is south of Nero. Oz is north of Nero. Pia is between Mira and "
     "Nero. Who is furthest north?", "Oz"),
    ("A recipe for 6 people needs 750 g of flour. You cook for 10 people but "
     "only have 1 kg of flour. How many grams short are you?", "250"),
    ("Every card with a vowel on one side has an even number on the other. "
     "You see cards showing: A, K, 4, 7. Which single card showing a letter "
     "must you turn over to test the rule?", "A"),
    ("A shop marks a coat up 50%, then discounts the marked price by 20%. "
     "The final price is 96 coins. What was the original price?", "80"),
    ("Three switches control three bulbs in a sealed room. You may enter the "
     "room only once. You flip switch one, wait ten minutes, turn it off, "
     "flip switch two, and enter. Which switch controls the bulb that is off "
     "but warm? Answer with a number.", "1"),
    ("Sam has 5 more marbles than Tao. Together they have 27. Tao then gives "
     "Sam 4 marbles. How many does Sam have now?", "20"),
    ("A ferry runs every 25 minutes starting 06:00. A passenger arrives at "
     "07:38. How many minutes must the passenger wait? ", "2"),
    ("All zibs are round. No round thing is heavy. Kel is heavy. Can Kel be "
     "a zib? Answer yes or no.", "no"),
    ("A field is 40 m by 25 m. A path 2 m wide runs along all four inside "
     "edges. What is the area in square metres of the grass inside the path?",
     "756"),
]
for i, (q, a) in enumerate(MULTIHOP):
    prompts.append({"id": 2000 + i, "domain": "multihop", "passage": q,
                    "task": "Think it through, then end your reply with a "
                            "final line of exactly this form:\nANSWER: <your "
                            "answer>",
                    "answer": a})

# ------------------------------------------------------------- B. constraint
# SIX simultaneous machine-checkable constraints. The easy set used one.
CONSTRAINT = [
    ("Write a product announcement for a waterproof notebook.",
     {"exact_lines": 4, "max_words": 90, "start_with": "Introducing",
      "must_include": ["waterproof", "notebook"], "forbid_words": ["amazing"],
      "end_with": "."}),
    ("Write instructions for changing a bicycle tyre.",
     {"exact_lines": 5, "max_words": 110, "start_with": "First",
      "must_include": ["tyre", "wheel"], "forbid_words": ["easy"],
      "end_with": "."}),
    ("Describe a thunderstorm for a weather report.",
     {"exact_lines": 3, "max_words": 70, "start_with": "Conditions",
      "must_include": ["lightning", "wind"], "forbid_words": ["scary"],
      "end_with": "."}),
    ("Write a short warning label for a hot beverage cup.",
     {"exact_lines": 3, "max_words": 45, "start_with": "Caution",
      "must_include": ["hot", "lid"], "forbid_words": ["sorry"],
      "end_with": "."}),
    ("Summarize the benefits of walking to work.",
     {"exact_lines": 4, "max_words": 85, "start_with": "Walking",
      "must_include": ["health", "cost"], "forbid_words": ["obviously"],
      "end_with": "."}),
    ("Write a museum placard for a Roman coin.",
     {"exact_lines": 3, "max_words": 75, "start_with": "This coin",
      "must_include": ["Roman", "bronze"], "forbid_words": ["priceless"],
      "end_with": "."}),
    ("Write onboarding steps for a new library card.",
     {"exact_lines": 5, "max_words": 100, "start_with": "Bring",
      "must_include": ["card", "library"], "forbid_words": ["simply"],
      "end_with": "."}),
    ("Write a note to a neighbour about a shared fence repair.",
     {"exact_lines": 4, "max_words": 90, "start_with": "Hello",
      "must_include": ["fence", "repair"], "forbid_words": ["urgent"],
      "end_with": "."}),
    ("Describe how to store fresh herbs.",
     {"exact_lines": 4, "max_words": 80, "start_with": "Trim",
      "must_include": ["water", "fridge"], "forbid_words": ["perfect"],
      "end_with": "."}),
    ("Write safety rules for a community swimming pool.",
     {"exact_lines": 5, "max_words": 95, "start_with": "Swimmers",
      "must_include": ["lifeguard", "diving"], "forbid_words": ["fun"],
      "end_with": "."}),
    ("Write a caption for a photograph of a harbour at dawn.",
     {"exact_lines": 3, "max_words": 60, "start_with": "At dawn",
      "must_include": ["harbour", "light"], "forbid_words": ["beautiful"],
      "end_with": "."}),
    ("Write setup steps for a two-person tent.",
     {"exact_lines": 5, "max_words": 105, "start_with": "Unpack",
      "must_include": ["poles", "stakes"], "forbid_words": ["quick"],
      "end_with": "."}),
    ("Write a short notice about a road closure.",
     {"exact_lines": 3, "max_words": 65, "start_with": "The road",
      "must_include": ["closed", "detour"], "forbid_words": ["unfortunately"],
      "end_with": "."}),
    ("Describe the rules of a simple card game.",
     {"exact_lines": 4, "max_words": 90, "start_with": "Deal",
      "must_include": ["deck", "player"], "forbid_words": ["basically"],
      "end_with": "."}),
    ("Write care instructions for a wool sweater.",
     {"exact_lines": 4, "max_words": 80, "start_with": "Wash",
      "must_include": ["wool", "dry"], "forbid_words": ["never ever"],
      "end_with": "."}),
    ("Write a brief agenda for a 30-minute team meeting.",
     {"exact_lines": 5, "max_words": 85, "start_with": "Open",
      "must_include": ["minutes", "actions"], "forbid_words": ["synergy"],
      "end_with": "."}),
    ("Describe a lighthouse for a travel guide.",
     {"exact_lines": 3, "max_words": 70, "start_with": "Standing",
      "must_include": ["lighthouse", "coast"], "forbid_words": ["stunning"],
      "end_with": "."}),
    ("Write steps for making cold-brew coffee.",
     {"exact_lines": 4, "max_words": 85, "start_with": "Grind",
      "must_include": ["coarse", "hours"], "forbid_words": ["delicious"],
      "end_with": "."}),
    ("Write a short policy note about quiet hours in a building.",
     {"exact_lines": 4, "max_words": 80, "start_with": "Residents",
      "must_include": ["quiet", "noise"], "forbid_words": ["please note"],
      "end_with": "."}),
    ("Describe how to plant a bare-root rose.",
     {"exact_lines": 5, "max_words": 100, "start_with": "Dig",
      "must_include": ["roots", "soil"], "forbid_words": ["lovely"],
      "end_with": "."}),
]
for i, (q, cons) in enumerate(CONSTRAINT):
    parts = [f"Answer with exactly {cons['exact_lines']} lines, one sentence "
             f"per line, no preamble, no bullets, no numbering.",
             f"Use at most {cons['max_words']} words in total.",
             f"The first line must begin with \"{cons['start_with']}\".",
             f"You must use the words: {', '.join(cons['must_include'])}.",
             f"You must NOT use the word(s): "
             f"{', '.join(cons['forbid_words'])}.",
             f"The final line must end with a full stop."]
    prompts.append({"id": 2100 + i, "domain": "constraint", "passage": q,
                    "task": " ".join(parts), "constraints": cons})

# ----------------------------------------------------------------- C. needle
# Distinctive planted facts at known depths in a long literary haystack.
corpus = (HERE / "referee" / "referee_corpus_literary.txt").read_text()
paras = [p for p in corpus.split("\n\n") if len(p) > 400]
rng = random.Random(20260819)
NEEDLES = [
    ("The harbourmaster's ledger recorded exactly {n} brass lanterns stored "
     "in the east vault.", "{n}", "How many brass lanterns were recorded in "
     "the east vault?"),
    ("Delphine kept her grandmother's recipe for {n} pickled walnuts in the "
     "blue tin.", "{n}", "How many pickled walnuts were in the recipe kept "
     "in the blue tin?"),
    ("The station clock at Ravensmoor had been stopped at {t} since the "
     "flood.", "{t}", "What time was the station clock at Ravensmoor "
     "stopped at?"),
    ("Only {n} copies of the Ashworth almanac survived the fire at the "
     "printing house.", "{n}", "How many copies of the Ashworth almanac "
     "survived the fire?"),
    ("The lighthouse keeper's cat was named {w} and slept on the log book.",
     "{w}", "What was the lighthouse keeper's cat named?"),
]
CATS = ["Perpetua", "Marmalade", "Thistle", "Bramwell", "Ozymandias"]
depths = [0.10, 0.30, 0.50, 0.70, 0.90]
k = 0
for d_i, depth in enumerate(depths):
    for n_i in range(2):                       # 2 needles per depth = 10
        tmpl, ans_t, question = NEEDLES[(d_i * 2 + n_i) % len(NEEDLES)]
        num = rng.randint(101, 989)
        time_v = f"{rng.randint(1, 12)}:{rng.choice(['07', '19', '23', '41'])}"
        word = CATS[(d_i * 2 + n_i) % len(CATS)]
        fact = tmpl.format(n=num, t=time_v, w=word)
        ans = ans_t.format(n=num, t=time_v, w=word)
        body = paras[k * 34:(k * 34) + 34]     # ~34 paragraphs -> ~6k tokens
        k += 1
        pos = max(1, int(len(body) * depth))
        body = body[:pos] + [fact] + body[pos:]
        prompts.append({"id": 2200 + d_i * 2 + n_i, "domain": "needle",
                        "passage": "\n\n".join(body),
                        "task": f"{question} Answer with the fact only, on a "
                                f"line of the form:\nANSWER: <value>",
                        "answer": ans, "depth": depth})

# ---------------------------------------------------------------- D. sustain
lit = json.load(open(HERE / "winrate" / "prompts.json"))
for i in range(10):
    prompts.append({"id": 2300 + i, "domain": "sustain",
                    "passage": lit[i * 6]["passage"],
                    "task": "Continue this passage for at least 800 words, "
                            "matching its voice, register and period. Do not "
                            "summarize, do not stop early, do not repeat "
                            "yourself. Output only the continuation."})

out = HERE / "winrate" / "prompts_capacity.json"
out.write_text(json.dumps(prompts, indent=1))
by = {}
for p in prompts:
    by[p["domain"]] = by.get(p["domain"], 0) + 1
print(f"{len(prompts)} prompts -> {out}")
print("  " + ", ".join(f"{k} {v}" for k, v in by.items()))
