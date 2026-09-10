#!/usr/bin/env python3
"""L9: 'Recent changes (YYYY-MM)' -> an accumulating '## Changelog' whose
entries are dated '### YYYY-MM — <label>'. Nothing is dropped; new releases
add an entry on top rather than overwriting the previous one.

The entry label comes from the block's own bold lead-in, so a repair stays a
repair and a refresh stays a refresh.
"""
import re, pathlib
D = pathlib.Path(__file__).parent / "drafts"

for p in sorted(D.glob("*.md")):
    t = p.read_text()
    m = re.search(r"^## Recent changes \((\d{4}-\d{2})\)\s*$\n(.*?)(?=^## |\Z)",
                  t, re.M | re.S)
    if not m:
        print(f"  {p.stem:32} — no prior entry; Changelog appears with the next one")
        continue
    date, block = m.group(1), m.group(2)

    lead = re.match(r"\s*\*\*(.+?)\*\*", block, re.S)
    label = "bundle refresh"
    if lead:
        raw = " ".join(lead.group(1).split())
        label = raw.rstrip(".").split(" — ")[0]
        label = label[0].lower() + label[1:] if label[:1].isupper() else label
        # keep the useful hook ("if mlx-lm gave you ModuleNotFoundError…")
        block = block[lead.end():].lstrip()
        if raw.rstrip(".") != label:
            block = f"**{raw}**\n\n{block}"

    entry = f"## Changelog\n\n### {date} — {label}\n\n{block.strip()}\n\n"
    t = t[:m.start()] + entry + t[m.end():]
    p.write_text(re.sub(r"\n{3,}", "\n\n", t))
    print(f"  {p.stem:32} ### {date} — {label}")
