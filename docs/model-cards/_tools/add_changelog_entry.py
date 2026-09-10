#!/usr/bin/env python3
"""Add a dated entry to the top of each card's Changelog. Nothing is replaced.

    python3 add_changelog_entry.py --date 2026-09-09 --label "runtime 0.2.0" \
        --body entry.md [--only Qwen3.8-27B] [--dry-run]

--body is a markdown file: the entry text, no heading (this script writes the
`### <date> — <label>` heading). Per-family text: run it once per --only.
Creates the `## Changelog` section if the card has none, placing it directly
after the intro (or after Requirements when present), which is where every
other card carries it.
"""
import argparse, re, pathlib, sys

ap = argparse.ArgumentParser()
ap.add_argument("--cards", default="cards", help="directory of card .md files")
ap.add_argument("--date", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--body", required=True, type=pathlib.Path)
ap.add_argument("--only", default="", help="only cards whose name starts with this")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a.date):
    sys.exit(f"--date must be YYYY-MM or YYYY-MM-DD, got {a.date!r}")
body = a.body.read_text().strip()
if body.lstrip().startswith("#"):
    sys.exit("--body must not contain a heading; this script writes it")

entry = f"### {a.date} — {a.label}\n\n{body}\n"
D = pathlib.Path(a.cards)
targets = sorted(p for p in D.glob("*.md") if p.stem.startswith(a.only))
if not targets:
    sys.exit(f"no cards match --only {a.only!r} in {D}/")

for p in targets:
    t = p.read_text()
    if f"### {a.date} — {a.label}" in t:
        print(f"  {p.stem:32} already has this entry — skipped")
        continue

    m = re.search(r"^## Changelog\s*$\n", t, re.M)
    if m:                                    # newest entry goes on top
        t = t[:m.end()] + "\n" + entry + "\n" + t[m.end():].lstrip("\n")
    else:                                    # create the section
        anchor = re.search(r"^## Requirements\s*$\n.*?(?=^## )", t, re.M | re.S) \
              or re.search(r"^## ", t, re.M)
        at = anchor.end()
        t = t[:at] + f"## Changelog\n\n{entry}\n" + t[at:]
    t = re.sub(r"\n{3,}", "\n\n", t)
    if a.dry_run:
        print(f"  {p.stem:32} would add ### {a.date} — {a.label}")
    else:
        p.write_text(t)
        print(f"  {p.stem:32} added ### {a.date} — {a.label}")
