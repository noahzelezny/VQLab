#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Assemble a LITERARY perplexity corpus from public-domain prose.

WHY THIS EXISTS. `referee_corpus.txt` is wikitext — encyclopedic register,
expository syntax, almost no narrative, metaphor, or irony. It measures
compression of factual prose, which is the wrong axis for a sidecar chosen
for storytelling and description. `referee_corpus_code.txt` covers the other
end. This fills the gap in the middle.

WHY PPL AND NOT MORE MULTIPLE-CHOICE. litbench (literary/) asks the model to
pick a correct reading, which is a technical proxy for something that isn't
technical, and buying sensitivity there costs 4x scoring time per model.
Perplexity over literary prose is continuous across thousands of tokens
rather than binary across hundreds of items: far more sensitive per unit of
compute for measuring what QUANTIZATION costs, and it makes no claim that a
passage has a correct reading. Division of labour:

    litbench          -> which MODEL   (comprehension, coarse, cheap)
    literary corpus   -> what the QUANT cost (compression, fine, cheap)
    referee_corpus    -> expository baseline (unchanged, still the ppl of record)

COPYRIGHT. Every source is public domain, fetched from Project Gutenberg by
id, never vendored into this repo. The manifest records title/author/PG id so
any number is reproducible and any claim about provenance is checkable.

MEMORIZATION — READ BEFORE CITING A CROSS-MODEL NUMBER. These are canonical
texts. Every candidate model has almost certainly seen them in training, so
ppl here is partly a memorization score, not purely a compression score.
That cuts two ways, and the split is exactly the division of labour above:

  VALID for quant ladders. Two quants of ONE model carry identical
  memorization, so it cancels; what remains is how much the quantizer
  degraded the model's grip on prose it knows. This is the intended use.

  CONFOUNDED across DIFFERENT models. gemma and Qwen have not memorized the
  same amount of Austen, so a cross-family ppl gap here cannot be separated
  from a difference in training exposure. Use litbench for that comparison,
  not this. If a cross-model literary ppl is ever needed, it needs held-out
  prose the models cannot have seen — which public-domain text cannot be.

SELECTION. Chosen for the qualities the sidecar is actually for — irony
(Austen, Wharton), metaphor and density (Conrad, Melville), subtext and
indirection (James, Joyce), sustained narrative voice (Eliot, Hardy).

    ./build_literary_corpus.py --plan       # print what it WOULD fetch, no network
    ./build_literary_corpus.py              # fetch + build
"""
import argparse
import json
import pathlib
import re
import sys
import urllib.request

HERE = pathlib.Path(__file__).parent
OUT = HERE / "referee_corpus_literary.txt"
MANIFEST = HERE / "referee_corpus_literary.manifest.json"

# (Project Gutenberg id, author, title, year, what it contributes)
WORKS = [
    (1342, "Austen",  "Pride and Prejudice",   1813, "free indirect style, sustained irony"),
    (158,  "Austen",  "Emma",                  1815, "irony, unreliable self-report"),
    (219,  "Conrad",  "Heart of Darkness",     1899, "metaphor, dense figurative prose"),
    (2701, "Melville", "Moby-Dick",            1851, "register shifts, extended metaphor"),
    (2814, "Joyce",   "Dubliners",             1914, "subtext, understatement, epiphany"),
    (209,  "James",   "The Turn of the Screw", 1898, "indirection, unreliable narration"),
    (541,  "Wharton", "The Age of Innocence",  1920, "social irony, subtext"),
    (145,  "Eliot",   "Middlemarch",           1871, "narrative voice, moral texture"),
    (110,  "Hardy",   "Tess of the d'Urbervilles", 1891, "description, narrative causality"),
    (1400, "Dickens", "Great Expectations",    1861, "first-person voice, retrospection"),
]

URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
ALT = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# Gutenberg wraps each work in a licence header/footer that is NOT the work
# and would pollute the corpus with boilerplate the model has memorized.
START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                   re.I | re.S)
END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                 re.I | re.S)


def strip_boilerplate(text):
    m = START.search(text)
    if m:
        text = text[m.end():]
    m = END.search(text)
    if m:
        text = text[:m.start()]
    # Transcriber notes, chapter rules, and the all-caps front matter add
    # register that is not the author's.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_prose(text):
    """Drop lines that are structural rather than prose."""
    keep = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            keep.append("")
            continue
        # chapter headings, roman numerals, all-caps runs
        if re.fullmatch(r"[IVXLC]+\.?", s):
            continue
        if re.fullmatch(r"(CHAPTER|BOOK|PART|VOLUME)\s+[\dIVXLC]+\.?", s, re.I):
            continue
        if len(s) > 3 and s == s.upper() and sum(c.isalpha() for c in s) > 3:
            continue
        keep.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()


def fetch(pg_id):
    last = None
    for tmpl in (URL, ALT):
        try:
            req = urllib.request.Request(
                tmpl.format(id=pg_id), headers={"User-Agent": "quantlab-referee/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:                            # noqa: BLE001
            last = e
    raise RuntimeError(f"PG {pg_id}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="print the source list and exit; touches no network")
    ap.add_argument("--per-work-chars", type=int, default=120_000,
                    help="characters taken from each work, from 15%% in "
                         "(skips front matter, lands in sustained narrative)")
    args = ap.parse_args()

    if args.plan:
        print(f"{len(WORKS)} public-domain works, "
              f"~{args.per_work_chars * len(WORKS) / 1e6:.1f}M chars target\n")
        for i, a, t, y, why in WORKS:
            print(f"  PG{i:<5} {a:<9} {t:<28} {y}   {why}")
        print(f"\nwould fetch from gutenberg.org and write {OUT.name}")
        return

    parts, manifest = [], []
    for pg_id, author, title, year, why in WORKS:
        print(f"fetching PG{pg_id} — {author}, {title} ...", flush=True)
        try:
            raw = fetch(pg_id)
        except RuntimeError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue
        body = clean_prose(strip_boilerplate(raw))
        start = int(len(body) * 0.15)
        chunk = body[start:start + args.per_work_chars]
        # end on a paragraph boundary so no passage is cut mid-sentence
        cut = chunk.rfind("\n\n")
        if cut > args.per_work_chars // 2:
            chunk = chunk[:cut]
        parts.append(chunk.strip())
        manifest.append({"pg_id": pg_id, "author": author, "title": title,
                         "year": year, "contributes": why, "chars": len(chunk)})
        print(f"  kept {len(chunk):,} chars")

    if not parts:
        raise SystemExit("nothing fetched — corpus not written")

    text = "\n\n".join(parts) + "\n"
    OUT.write_text(text)
    MANIFEST.write_text(json.dumps(
        {"works": manifest, "total_chars": len(text),
         "note": "public domain, fetched from Project Gutenberg by id; "
                 "text is not vendored into this repo"}, indent=1))

    print(f"\nwrote {OUT}  ({len(text):,} chars, ~{len(text) // 4:,} tokens est.)")
    print(f"wrote {MANIFEST}")
    print("\nNOTE: referee/score_streaming.py scores an 8192-token PREFIX by "
          "default. This corpus is deliberately much larger so the same file "
          "supports several disjoint windows — raise --max-tokens, or score "
          "windows separately, if you want more than one sample per model.")


if __name__ == "__main__":
    main()
