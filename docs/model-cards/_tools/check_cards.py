#!/usr/bin/env python3
"""Pre-publish lint. Run before copying a card to its artifact README.

    python3 _tools/check_cards.py .
"""
import re, sys, glob, pathlib, yaml

U = ["Measured results", "Run it", "Methodology", "Verification",
     "Limitations", "Paper", "Provenance"]
D = sys.argv[1] if len(sys.argv) > 1 else "."
fails = []

for f in sorted(glob.glob(f"{D}/*.md")):
    p = pathlib.Path(f); n = p.stem
    if n in ("README", "entry"):
        continue
    t = p.read_text()
    def bad(msg): fails.append(f"{n}: {msg}")

    m = re.match(r"^---\n(.*?)\n---\n", t, re.S)
    if not m: bad("no frontmatter"); continue
    try: yaml.safe_load(m.group(1))
    except Exception as e: bad(f"bad YAML ({e})")

    h1 = re.search(r"^# (.+)$", t, re.M)
    if not h1 or h1.group(1) != n: bad("H1 is not the bare repo name")

    heads = [x.strip() for x in re.findall(r"^## (.+)$", t, re.M)]
    for u in U:
        if u not in heads: bad(f"missing section: {u}")

    if "inherits that licence" not in t: bad("no licence statement")
    if "Quantization: TheDrainFlorist" not in t: bad("no attribution")

    # strikethrough hazards: `~` and `~~` outside code render as strikethrough
    fence = False
    for i, line in enumerate(t.split("\n"), 1):
        if line.lstrip().startswith("```"): fence = not fence; continue
        if fence: continue
        clean = re.sub(r"`[^`]*`", "", line)
        if "~" in clean:
            bad(f"line {i}: '~' renders as strikethrough — use ≈ or drop it")

    if t.count("```") % 2: bad("unbalanced code fence")
    for marker in ("TODO", "TO MEASURE", "[PENDING", "FIXME"):
        if marker in t: bad(f"placeholder marker: {marker}")
    if "zenodo.22136000" in t: bad("version DOI — use concept DOI 22119017")

print("\n".join(f"  {x}" for x in fails) if fails
      else f"  {len(glob.glob(f'{D}/*.md')) - 1} cards clean")
sys.exit(1 if fails else 0)
