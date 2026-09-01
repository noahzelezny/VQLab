#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Build winrate/prompts_domains.json — the 3 non-literary lenses for the
gemma-small publish decision (E56). litbench is one narrow instrument; the
sidecar also does instruction-following, summarization, and dialogue. 60
prompts, deterministic, same {id, passage, task} shape winrate_bench eats.

The 20 instruction-following prompts carry MACHINE-CHECKABLE constraints
(word caps, mandated openings, exact counts) so check_constraints.py can
score them pass/fail with n-scale error bars and no judge chip at all.
IDs are namespaced 1000+ so they can never collide with the literary set.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
lit = json.load(open(HERE / "winrate" / "prompts.json"))

prompts = []

# --- instruction-following: verifiable constraints (ids 1000-1019) --------
IF_TASKS = [
    ("Explain why the sky is blue.", {"max_words": 60}),
    ("Describe how yeast makes bread rise.", {"max_words": 80}),
    ("Give advice for a first job interview.", {"start_with": "First,"}),
    ("Explain what a checksum is.", {"start_with": "A checksum"}),
    ("List exactly 5 uses for a brick.", {"exact_lines": 5}),
    ("List exactly 3 famous rivers.", {"exact_lines": 3}),
    ("Explain photosynthesis without using the letter 'e'.", {"forbid_char": "e"}),
    ("Write a two-sentence horror story.", {"max_sentences": 2}),
    ("Define recursion in one sentence.", {"max_sentences": 1}),
    ("Explain inflation to a child.", {"max_words": 70}),
    ("Give exactly 4 tips for better sleep.", {"exact_lines": 4}),
    ("Describe the water cycle.", {"start_with": "Water"}),
    ("Explain what an API is.", {"max_words": 50}),
    ("Write a haiku about autumn (3 lines).", {"exact_lines": 3}),
    ("List exactly 6 primary and secondary colors.", {"exact_lines": 6}),
    ("Explain gravity in exactly two sentences.", {"max_sentences": 2}),
    ("Give directions for boiling an egg.", {"start_with": "Place"}),
    ("Summarize the plot of Cinderella.", {"max_words": 90}),
    ("Explain what a firewall does.", {"max_words": 60}),
    ("List exactly 5 mammals that can swim.", {"exact_lines": 5}),
]
for i, (q, cons) in enumerate(IF_TASKS):
    parts = []
    if "max_words" in cons:
        parts.append(f"Use at most {cons['max_words']} words.")
    if "max_sentences" in cons:
        parts.append(f"Use at most {cons['max_sentences']} sentence(s).")
    if "exact_lines" in cons:
        parts.append(f"Answer with exactly {cons['exact_lines']} lines, "
                     "one item per line, no preamble.")
    if "start_with" in cons:
        parts.append(f"Your answer must begin with the word(s) "
                     f"\"{cons['start_with']}\".")
    if "forbid_char" in cons:
        parts.append(f"Do not use the letter '{cons['forbid_char']}' "
                     "anywhere in your answer.")
    prompts.append({"id": 1000 + i, "passage": q,
                    "task": " ".join(parts), "constraints": cons,
                    "domain": "instruct"})

# --- summarization: reuse literary passages 0-19 (ids 1100-1119) ----------
for i in range(20):
    prompts.append({"id": 1100 + i, "passage": lit[i]["passage"],
                    "task": "Summarize this passage in 2-3 sentences. "
                            "Capture who is involved and what changes. "
                            "Output only the summary.",
                    "domain": "summar"})

# --- dialogue: multi-turn-flavored single prompts (ids 1200-1219) ---------
DIALOG = [
    "My landlord raised rent 15% with 20 days notice. What should I ask him first?",
    "I burned the garlic twice tonight. What am I doing wrong?",
    "My 8-year-old asked why the moon follows the car. How do I answer her?",
    "A coworker keeps taking credit for my ideas in meetings. Help me plan what to say.",
    "I have $60 and four dinner guests on Friday. Sketch me a menu.",
    "My sourdough starter smells like nail polish. Is it dead?",
    "I want to start running but my knees hurt after a mile. What now?",
    "My cat wakes me at 4am every day. How do I make it stop?",
    "I froze during my presentation today. How do I recover with my team tomorrow?",
    "My tomato leaves are curling and yellow at the edges. Diagnose it.",
    "Should I tell my friend her wedding band was out of tune? She asked how it was.",
    "I've reread the same page four times tonight. How do people actually focus?",
    "My teenager wants to quit piano after 6 years. Do I let him?",
    "The mechanic quoted $900 for brakes. How do I tell if that's fair?",
    "I said something dumb in the group chat and nobody replied. Damage control?",
    "My houseplant drops one leaf a day. What questions would you ask me?",
    "First time hosting Thanksgiving. What do people always forget?",
    "My neighbor's dog barks from 9 to 11 every night. What's my first move?",
    "I keep snoozing through morning workouts. Redesign my evening for me.",
    "Grandma's recipe says 'bake until done.' It's a custard pie. Help.",
]
for i, q in enumerate(DIALOG):
    prompts.append({"id": 1200 + i, "passage": q,
                    "task": "Reply as a helpful conversation partner in "
                            "under 150 words. Be concrete and specific.",
                    "domain": "dialog"})

out = HERE / "winrate" / "prompts_domains.json"
out.write_text(json.dumps(prompts, indent=1))
print(f"{len(prompts)} prompts -> {out}")
