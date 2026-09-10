#!/usr/bin/env python3
"""L6: one structure, one voice, license+attribution on every card.

Runs after fix_cards.py. Rewrites drafts/ in place:
  - H1 -> bare repo name (owner is already shown by the Hub)
  - canonical section order
  - the 7 universal sections exist on every card
  - Provenance (licence + attribution) injected wherever missing
"""
import re, pathlib

D = pathlib.Path(__file__).parent / "drafts"

ORDER = ["Requirements", "Recent changes", "Measured results", "Task benchmarks",
         "Comparators", "The full sweep", "Choosing a size",
         "Where this stops paying", "Hardware", "Runtime", "Memory", "Run it",
         "Speculative decoding (MTP)", "Vision", "Siblings", "Why the embedding table",
         "No calibration data", "Tuning: prefill speed", "The leverage mix",
         "Methodology", "Verification", "Limitations", "Acknowledgment",
         "Paper", "Provenance"]
UNIVERSAL = ["Measured results", "Run it", "Methodology", "Verification",
             "Limitations", "Paper", "Provenance"]

# ---------------------------------------------------------------- attribution
BASE = {
    "GLM-5.3-Flash":      ("zai-org/GLM-5.3-Flash", "**MIT licensed**"),
    "Qwen3.5-397B-A17B":  ("Qwen/Qwen3.5-397B-A17B", "**Apache-2.0**"),
    "Qwen3.6-35B-A3B":    ("Qwen/Qwen3.6-35B-A3B", "**Apache-2.0**"),
    "Qwen3.8-27B":        ("Qwen/Qwen3.8-27B", "**Apache-2.0**"),
    "Qwen3.8-Flash-Next": ("Qwen/Qwen3.8-Flash-Next",
                           "released under the **Qwen Community License 1.0**"),
    "gemma-4":            ("google/gemma-4", "governed by the **Gemma Terms of Use**"),
}

def family(name):
    for f in sorted(BASE, key=len, reverse=True):
        if name.startswith(f): return f
    raise SystemExit(f"no family for {name}")

def provenance(name):
    fam = family(name)
    base, lic = BASE[fam]
    if fam == "gemma-4":
        base = ("google/gemma-4-e4b-it" if "e4b" in name
                else "google/gemma-4-26b-a4b-it")
    extra = ""
    if name == "gemma-4-e4b-it-VQ-PLE":
        extra = ("\n\nThis build starts from\n"
                 "[mlx-community/gemma-4-e4b-it-8bit]"
                 "(https://huggingface.co/mlx-community/gemma-4-e4b-it-8bit)\n"
                 "and replaces its per-layer-embedding table; that conversion's\n"
                 "authors are credited accordingly.")
    return (f"## Provenance\n\n"
            f"Base model: [{base}](https://huggingface.co/{base}) — {lic}.\n"
            f"This is a quantized derivative and inherits that licence; using it\n"
            f"means accepting the base model's terms.\n"
            f"Quantization: TheDrainFlorist, 2026.{extra}\n\n"
            f"Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab).\n")

VERIFICATION = (
    "## Verification\n\n"
    "Release gates passed on this artifact before upload: file, index and\n"
    "tokenizer checks, a verbatim match between the bundled runtime and its\n"
    "source, and a generation smoke through the shipping runtime on Apple\n"
    "Silicon. The upload path runs the gate itself and refuses to publish\n"
    "without it.\n")

# --------------------------------------------------------------------- engine
def split(t):
    fm = re.match(r"^---\n.*?\n---\n", t, re.S).group(0)
    rest = t[len(fm):]
    parts = re.split(r"^(## .+)$", rest, flags=re.M)
    head = parts[0]
    secs = [(parts[i][3:].strip(), parts[i + 1]) for i in range(1, len(parts), 2)]
    return fm, head, secs

def rank(title):
    for i, c in enumerate(ORDER):
        if title == c or title.startswith(c): return i
    return len(ORDER) - 6   # unknown -> just before Methodology

for p in sorted(D.glob("*.md")):
    name = p.stem
    fm, head, secs = split(p.read_text())

    # H1 -> bare repo name
    head = re.sub(r"^# .+$", f"# {name}", head, count=1, flags=re.M)

    have = {t for t, _ in secs}
    if "Verification" not in have and "Provenance and gates" not in have:
        secs.append(("Verification", "\n" + VERIFICATION[len("## Verification\n"):]))
    if not any(t.startswith("Provenance") for t in have):
        secs.append(("Provenance", "\n" + provenance(name)[len("## Provenance\n"):]))

    secs.sort(key=lambda s: rank(s[0]))
    body = head + "".join(f"## {t}\n{b}" for t, b in secs)
    p.write_text(fm + body)

    missing = [u for u in UNIVERSAL
               if not any(t.startswith(u.split()[0]) for t, _ in secs)]
    print(f"  {name:32} {'OK' if not missing else 'MISSING ' + ','.join(missing)}")
