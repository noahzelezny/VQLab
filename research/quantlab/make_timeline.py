#!/usr/bin/env python3
"""Extract TIMELINE.md -- every experiment, in order, and what it settled.

    ./make_timeline.py

Read-only over EXPERIMENTS.md. This is receipt material: if the model
artifacts are deleted, this is the document that shows the work was done, in
what order, and what each step decided.

It EXTRACTS rather than summarizes. Every line is the experiment's own heading
text, not a paraphrase -- a summary written now could smuggle in today's
beliefs about what an old experiment showed, which is exactly the failure the
log's retraction discipline exists to prevent. Dates come from the heading or from the first date-stamped line inside the
entry, IGNORING later editorial annotations (a ">" note saying "SUFFIXED
2026-08-25" records when someone amended the entry, not when the experiment
ran). Where an entry carries no date of its own, the previous entry's date is
carried forward and marked "~" -- the log is chronological, so the ORDER is
exact even where the date is inferred. A "~" date can be off by days; treat it
as placement, not provenance.

PRE-REGISTRATION and RESULT entries under one E-number are paired on one row,
so a reader can see at a glance which predictions were made before the numbers
existed -- and which pre-registrations never got a result.
"""
import re, collections, datetime, sys

SRC = "EXPERIMENTS.md"
H = re.compile(r"^##\s+(E\d+[A-Za-z0-9\-]*)\s*(?:—|--|-)\s*(.*)$")
BULLET = re.compile(r"^- \*\*(E\d+)\s*\(([^)]+)\)\s*:?\s*\*\*:?\s*(.*)$")
DATE_IN = re.compile(r"\(?(\d{2}-\d{2})(?:/\d{2})?\)?")
DATE_LINE = re.compile(r"(?:^|\s)20?26-(\d{2}-\d{2})")

KIND = [("PRE-REGISTRATION", "pre-reg"), ("PRE-REGISTERED", "pre-reg"),
        ("RETRACTION", "retraction"), ("RESULT", "result"),
        ("PARTIAL", "partial"), ("STUB", "stub"),
        ("ADDENDUM", "addendum"), ("INVESTIGATION", "investigation")]


def kind_of(title):
    up = title.upper()
    for needle, k in KIND:
        if up.startswith(needle) or up.startswith("M3 — " + needle) or needle in up[:32]:
            return k
    return "entry"


def strip_kind(title):
    return re.sub(r"^(M[34]\s*(—|-)\s*)?[A-Z\- ]{4,20}:\s*", "", title).strip()


def main():
    lines = open(SRC, encoding="utf-8").read().split("\n")
    ents = collections.OrderedDict()

    for i, ln in enumerate(lines):
        b = BULLET.match(ln)
        if b:
            e, d, txt = b.group(1), b.group(2), b.group(3)
            ents.setdefault(e, {"rows": [], "date": d, "line": i + 1})
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip():
                txt += " " + lines[j].strip(); j += 1
            ents[e]["rows"].append(("entry", " ".join(txt.split()).rstrip(" —-"), i + 1))
            continue
        m = H.match(ln)
        if not m:
            continue
        e, title = m.group(1), m.group(2).strip()
        d = None
        dm = DATE_IN.search(title)
        if dm:
            d = dm.group(1)
        if not d:
            for j in range(i + 1, min(i + 12, len(lines))):
                # ">" lines are later editorial notes (supersession, suffix
                # fixes). Their dates are when SOMEONE ANNOTATED the entry, not
                # when the experiment ran -- reading them was putting 08-25 on
                # experiments from 08-23.
                if lines[j].lstrip().startswith(">"):
                    continue
                dl = DATE_LINE.search(lines[j])
                if dl:
                    d = dl.group(1); break
        ents.setdefault(e, {"rows": [], "date": None, "line": i + 1})
        if d and not ents[e]["date"]:
            ents[e]["date"] = d
        ents[e]["rows"].append((kind_of(title), strip_kind(title), i + 1))

    # Entries with no date of their own sit between dated neighbours in a
    # chronological log. Carry the last known date forward, marked "~" so an
    # inferred date is never mistaken for a recorded one.
    last = None
    for e in sorted(ents, key=lambda x: ents[x]["line"]):
        if ents[e]["date"]:
            last = ents[e]["date"]
        elif last:
            ents[e]["date"] = "~" + last

    def enum(e):
        return int(re.sub(r"\D", "", e) or 0)

    order = sorted(ents, key=lambda e: (enum(e), e))

    out = ["# Experiment timeline", "",
           "Every experiment in this project, in order, with what each one settled.",
           "Extracted verbatim from `EXPERIMENTS.md` headings by `make_timeline.py` --",
           "no paraphrase, so nothing here can drift from the log it came from.", "",
           "**This is the receipt.** The model artifacts are large and most are",
           "deletable; the finding, its date, and its line in the log are not. For any",
           "row, `EXPERIMENTS.md:<line>` has the full entry with metric values,",
           "instrument, and conditions.", "",
           "Dates marked `~` are inferred from position in the log, not recorded in",
           "the entry, and can be off by days. The ORDER is exact regardless; where a",
           "date matters, `paper/LEDGER.md` carries dated entries for the published",
           "results.", "",
           "`pre-reg` marks a prediction registered BEFORE the numbers existed. A",
           "pre-reg with no matching result is shown as such rather than hidden --",
           "those are the questions this project asked and did not answer.", ""]

    npre = nres = nopen = 0
    out.append("| # | date | kind | what it settled | log |")
    out.append("|---|---|---|---|---|")
    for e in order:
        v = ents[e]
        kinds = {k for k, _, _ in v["rows"]}
        has_res = bool(kinds & {"result", "partial", "retraction", "entry", "stub"})
        if "pre-reg" in kinds:
            npre += 1
            if not has_res:
                nopen += 1
        if "result" in kinds:
            nres += 1
        for n, (k, title, ln) in enumerate(v["rows"]):
            title = title.replace("|", "\\|")
            if len(title) > 150:
                title = title[:147] + "..."
            open_mark = " **(no result)**" if (k == "pre-reg" and not has_res) else ""
            out.append("| {e} | {d} | {k} | {t}{o} | [{ln}](EXPERIMENTS.md#L{ln}) |".format(
                e=e if n == 0 else "", d=(v["date"] or "--") if n == 0 else "",
                k=k, t=title, o=open_mark, ln=ln))
    out.insert(11, f"**{len(order)} experiments, {nres} with a written result, "
                   f"{npre} pre-registered, {nopen} pre-registered and never closed.**\n")
    open("TIMELINE.md", "w").write("\n".join(out) + "\n")
    print(f"wrote TIMELINE.md — {len(order)} experiments, {nres} results, "
          f"{npre} pre-registered, {nopen} left open")


if __name__ == "__main__":
    main()
