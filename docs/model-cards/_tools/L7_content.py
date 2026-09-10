#!/usr/bin/env python3
"""L7: the moves that need content shifted, not headings renamed.

  - 'Provenance and gates' -> 'Verification' + 'Provenance'   (5 cards)
  - Flash-Next: Run it, Methodology (from 'The leverage mix'), Limitations
  - e4b: Methodology
  - GLM: a '## Paper' section (their DOI was inline prose only)
Run after restructure.py.
"""
import re, pathlib
D = pathlib.Path(__file__).parent / "drafts"

GATES = ("Release gates passed on this artifact before upload: file, index and\n"
         "tokenizer checks, a verbatim match between the bundled runtime and its\n"
         "source, and a generation smoke through the shipping runtime on Apple\n"
         "Silicon. The upload path runs the gate itself and refuses to publish\n"
         "without it.\n")

PAPER = """## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

"""

FIT = ("Fitted **data-free** from the bf16 checkpoint — k-means / Lloyd over weight\n"
       "subvectors, seed 1234, no Hessian, no activation statistics, no calibration\n"
       "corpus. Recipes are in the VQLab repo.\n\n")

def secs_of(t):
    fm = re.match(r"^---\n.*?\n---\n", t, re.S).group(0)
    rest = t[len(fm):]
    parts = re.split(r"^(## .+)$", rest, flags=re.M)
    return fm, parts[0], [[parts[i][3:].strip(), parts[i+1]] for i in range(1, len(parts), 2)]

def rebuild(fm, head, secs):
    return fm + head + "".join(f"## {t}\n{b}" for t, b in secs)

def find(secs, title):
    for s in secs:
        if s[0] == title: return s
    return None

for p in sorted(D.glob("*.md")):
    name = p.stem
    fm, head, secs = secs_of(p.read_text())
    titles = [s[0] for s in secs]

    # 1. split 'Provenance and gates'
    pg = find(secs, "Provenance and gates")
    if pg:
        txt = pg[1]
        m = re.search(r"Release gates passed[^.]*\.(?:[^.]*\.)*?Apple Silicon\.", txt, re.S)
        rest = (txt.replace(m.group(0), "").strip() if m else txt.strip())
        rest = re.sub(r"\n{3,}", "\n\n", rest)
        pg[0], pg[1] = "Verification", "\n" + GATES + "\n"
        secs.append(["Provenance", "\n" + rest + "\n\n"])

    # 2. 'The leverage mix' -> Methodology, with the fitting para in front
    lm = find(secs, "The leverage mix")
    if lm:
        lm[0] = "Methodology"
        lm[1] = "\n" + FIT + lm[1].strip() + "\n\n"

    # 3. Flash-Next: a real 'Run it'
    if name.startswith("Qwen3.8-Flash-Next") and not find(secs, "Run it"):
        secs.append(["Run it", f"""
```bash
pip install git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1788/head

python -m mlx_lm generate \\
  --model TheDrainFlorist/{name} \\
  --prompt "Explain vector quantization briefly." \\
  --max-tokens 512
```

The VQ runtime needs no patches — it ships inside the checkpoint as
`model.py` and stock `mlx-lm` executes it. The PR above is required only for
the base architecture; see Requirements.

"""])

    # 4. Flash-Next: Limitations, built only from facts already on the card
    if name.startswith("Qwen3.8-Flash-Next") and not find(secs, "Limitations"):
        exo = ("- **Through-exo throughput is measured on the 2.1bpw rung only**\n"
               "  (23.7–25.5 tok/s, acceptance 0.82–0.88). The sidecar head is the\n"
               "  same file on every rung, but this rung's cluster numbers are not\n"
               "  measured.\n") if "2.1bpw" not in name else ""
        secs.append(["Limitations", f"""
- **This model needs an unreleased `mlx-lm`.** The `qwen4_exp` architecture is
  in [PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788), still unmerged
  as of 2026-09-09. See Requirements.
- **Stock exo cannot serve it** — upstream pins a released `mlx-lm` that lacks
  the architecture. Use our fork's `mtp-stage1` branch.
{exo}- **Rank the table by KL, not perplexity.** Perplexity is an aggregate over
  finite text and absorbs offsetting errors; several rungs read within noise of
  the teacher on perplexity while differing by an order of magnitude in KL.
- **The affine comparators are our own conversions**, not community builds.
- **Perplexity is not comparable across model families** — only within this
  table, which is one instrument on one corpus set.

"""])

    # 5. e4b: Methodology
    if name == "gemma-4-e4b-it-VQ-PLE" and not find(secs, "Methodology"):
        secs.append(["Methodology", "\n" + FIT +
            "Only the per-layer-embedding table is replaced; every other tensor is\n"
            "carried over from the 8-bit conversion unchanged. See *Why the embedding\n"
            "table* above for why that is the high-leverage target.\n\n"])

    # 6. GLM: a Paper section (the DOI was inline prose only)
    if not find(secs, "Paper"):
        secs.append(["Paper", PAPER[len("## Paper\n"):]])


    # 7. GLM: split the generate call out of Requirements into 'Run it'
    if name.startswith("GLM") and not find(secs, "Run it"):
        req = find(secs, "Requirements")
        blk = re.search(r"```bash\n.*?```\n", req[1], re.S)
        if blk:
            req[1] = req[1].replace(blk.group(0),
                                    "```bash\npip install 'mlx-vlm>=0.6.17'\n```\n")
        secs.append(["Run it", f"""
```bash
python -m mlx_vlm generate \\
  --model TheDrainFlorist/{name} \\
  --prompt "Explain vector quantization briefly." \\
  --max-tokens 512
```

The VQ runtime needs no patches — it ships inside the checkpoint as
`model.py`, declared via `model_file` in `config.json`, and resolves under
both `mlx-lm` and `mlx_vlm`.

"""])

    from importlib import import_module
    ORDER = import_module("restructure").ORDER
    def rank(t):
        for i, c in enumerate(ORDER):
            if t == c or t.startswith(c): return i
        return len(ORDER) - 6
    secs.sort(key=lambda s: rank(s[0]))
    p.write_text(rebuild(fm, head, secs))

print("L7 applied")
