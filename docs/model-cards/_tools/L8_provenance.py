#!/usr/bin/env python3
"""L8: one canonical Provenance on all 20 — licence + attribution, same voice.

Preserves the two card-specific things worth keeping: GLM's upstream citation
request (they ask for it) and the internal `Local artifact:` breadcrumb.
"""
import re, pathlib
from restructure import BASE, family, D

for p in sorted(D.glob("*.md")):
    name = p.stem
    t = p.read_text()
    m = re.search(r"^## Provenance\s*$\n(.*?)(?=^## |\Z)", t, re.M | re.S)
    old = m.group(1) if m else ""

    # keep: upstream citation request (GLM) and the local-artifact breadcrumb
    bib = re.search(r"The upstream authors.*?```bibtex.*?```", old, re.S)
    art = re.search(r"Local artifact: `[^`]+`\.", old)

    fam = family(name)
    base, lic = BASE[fam]
    if fam == "gemma-4":
        base = "google/gemma-4-e4b-it" if "e4b" in name else "google/gemma-4-26b-a4b-it"

    block = (f"Base model: [{base}](https://huggingface.co/{base}) — {lic}.\n"
             f"This is a quantized derivative and inherits that licence; using it\n"
             f"means accepting the base model's terms.\n"
             f"Quantization: TheDrainFlorist, 2026.\n")

    if name == "gemma-4-e4b-it-VQ-PLE":
        block += ("\nThis build starts from\n"
                  "[mlx-community/gemma-4-e4b-it-8bit]"
                  "(https://huggingface.co/mlx-community/gemma-4-e4b-it-8bit)\n"
                  "and replaces its per-layer-embedding table; that conversion's\n"
                  "authors are credited accordingly.\n")
    if bib:
        block += "\n" + bib.group(0).strip() + "\n"
    if art:
        block += "\n" + art.group(0) + "\n"
    block += "\nBuilt with MLX and [VQLab](https://github.com/noahzelezny/VQLab).\n"

    new = f"## Provenance\n\n{block}"
    t = (t[:m.start()] + new if m else t.rstrip() + "\n\n" + new)
    t = re.sub(r"\n{3,}", "\n\n", t)          # collapse the split artefacts
    p.write_text(t.rstrip() + "\n")
    print(f"  {name:32} bibtex={'Y' if bib else '-'} artifact={'Y' if art else '-'}")
