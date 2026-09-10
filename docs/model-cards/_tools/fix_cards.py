#!/usr/bin/env python3
"""Layered corrections for the 20 TheDrainFlorist VQ model cards.

Each layer is independently toggleable:  python3 fix_cards.py L1 L2 L3 L4
Source  : drafts/<repo>.md   (staged byte-identical from live)
Output  : rewritten in place; run `make_diffs.sh` to review.
"""
import re, sys, pathlib

D = pathlib.Path(__file__).parent / "drafts"
LAYERS = set(sys.argv[1:]) or {"L1", "L2", "L3", "L4", "L5"}
OWNER = "TheDrainFlorist"

FAMILY_TAG = {
    "GLM-5.3-Flash": "glm", "Qwen3.5-397B-A17B": "qwen3.5",
    "Qwen3.6-35B-A3B": "qwen3.6", "Qwen3.8-27B": "qwen3.8",
    "Qwen3.8-Flash-Next": "qwen3.8", "gemma-4": "gemma-4",
}
MULTIMODAL = ("GLM-5.3-Flash", "gemma-4", "Qwen3.6-35B-A3B")


def repo_of(p):       return p.stem
def family_of(name):
    for f in FAMILY_TAG:
        if name.startswith(f): return f
    raise SystemExit(f"unknown family: {name}")


def split_fm(t):
    m = re.match(r"^---\n(.*?)\n---\n", t, re.S)
    return m.group(1), t[m.end():]


def join_fm(fm, body):  return f"---\n{fm}\n---\n{body}"


# ---------------------------------------------------------------- L1 blockers
GLM_RUN = """
```bash
pip install mlx-vlm

python -m mlx_vlm generate \\
  --model {owner}/{repo} \\
  --prompt "Explain vector quantization briefly." \\
  --max-tokens 512
```
"""

def l1_glm_run_command(name, fm, body):
    """3.1bpw and 3.6bpw ship a bare `pip install mlx-vlm` with no generate
    line — the only two cards in the lineup you cannot copy-paste from."""
    if not name.startswith("GLM") or "mlx_vlm generate" in body:
        return fm, body
    block = GLM_RUN.format(owner=OWNER, repo=name).strip()
    return fm, body.replace("```bash\npip install mlx-vlm\n```", block, 1)


def l1_strip_todo(name, fm, body):
    """`[TO MEASURE: ...]` is shipped live on two GLM cards."""
    return fm, re.sub(
        r"^- \[TO MEASURE:[^\]]*\]\n", 
        "- Single-box peak memory is not measured for this rung: it does not\n"
        "  fit one machine. Cluster figures are above.\n",
        body, flags=re.M)


# ---------------------------------------------------------------- L2 metadata
def l2_title(name, fm, body):
    """Three H1 conventions across 20 cards, and two H1s name the wrong repo
    (397B-3.1bpw says 'VQ-3bpw'; gemma-26b says 'VQ-d2K2048')."""
    return fm, re.sub(r"^# .+$", f"# {OWNER}/{name}", body, count=1, flags=re.M)


def l2_tags(name, fm, body):
    """gemma-4-e4b-it-VQ-PLE carries only ['mlx']; family tags survive on
    gemma-26b alone. Normalise every card to the full set."""
    fam = family_of(name)
    tags = ["mlx", "quantized", "vector-quantization", "apple-silicon",
            FAMILY_TAG[fam]]
    if name.startswith(MULTIMODAL): tags.append("multimodal")
    block = "tags:\n" + "".join(f"- {t}\n" for t in dict.fromkeys(tags))
    fm2 = re.sub(r"^tags:\n(?:- .*\n)+", block, fm + "\n", flags=re.M).rstrip()
    return fm2, body


def l2_e4b_size(name, fm, body):
    """Normalising e4b's free-form H1 dropped its size; house style leads the
    subtitle with it."""
    if name != "gemma-4-e4b-it-VQ-PLE": return fm, body
    return fm, body.replace(
        "**The community 8-bit, with a better embedding table.**",
        "**7.39 GiB — the community 8-bit, with a better embedding table.**", 1)


# ------------------------------------------------------------------- L3 paper
PAPER = """
## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.{doi}) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/{owner}/below-six-bits)
"""

def l3_paper(name, fm, body, doi="22119017"):
    """17 of 20 cards have no paper link at all. Insert before Provenance,
    else append."""
    if "zenodo" in body.lower(): return fm, body
    block = PAPER.format(doi=doi, owner=OWNER)
    m = re.search(r"^## Provenance", body, re.M)
    if m: return fm, body[:m.start()] + block.lstrip() + "\n" + body[m.start():]
    return fm, body.rstrip() + "\n" + block


# ----------------------------------------------------------------- L4 runtime
def l4_pin_versions(name, fm, body):
    """Every card says bare `pip install mlx-lm`. Pin a floor so tonight's
    runtime has a stated minimum."""
    body = body.replace("pip install mlx-lm\n", "pip install 'mlx-lm>=0.31.3'\n")
    body = body.replace("pip install mlx-vlm\n", "pip install 'mlx-vlm>=0.6.17'\n")
    return fm, body


def l4_date_hedge(name, fm, body):
    """'still unmerged at the time of writing' rots. PR 1788 confirmed OPEN
    as of 2026-09-09 — claim stays true, but date it."""
    return fm, body.replace(
        "still unmerged at the time of writing.",
        "still unmerged as of 2026-09-09.")



# -------------------------------------------------------------- L5 vocabulary
HEADING_CANON = [
    ("How it was built",                        "Methodology"),
    ("Known limitations",                       "Limitations"),
    ("Requirements — read before downloading",  "Requirements"),
    ("Memory, measured externally",             "Memory"),
    ("Speculative decoding (MTP) — sidecar included",
                                                "Speculative decoding (MTP)"),
    ("Speculative decoding (MTP) — optional sidecar",
                                                "Speculative decoding (MTP)"),
]

def l5_headings(name, fm, body):
    """One name per concept. 'Methodology' is canon over 'How it was built';
    'Limitations' over 'Known limitations'; the MTP section had two names."""
    for old, new in HEADING_CANON:
        body = re.sub(rf"^## {re.escape(old)}\s*$", f"## {new}", body, flags=re.M)
    return fm, body


def l5_fix_glm_doi(name, fm, body):
    """The three GLM cards inline the *version* DOI (22136000). Model cards
    take the concept DOI so they never go stale against a new paper version."""
    return fm, body.replace("zenodo.22136000", "zenodo.22119017")


REGISTRY = {
    "L1": [l1_glm_run_command, l1_strip_todo],
    "L2": [l2_title, l2_tags, l2_e4b_size],
    "L3": [l3_paper],
    "L4": [l4_pin_versions, l4_date_hedge],
    "L5": [l5_headings, l5_fix_glm_doi],
}

changed = 0
for p in sorted(D.glob("*.md")):
    name = repo_of(p)
    orig = p.read_text()
    fm, body = split_fm(orig)
    for layer in ("L1", "L2", "L3", "L4", "L5"):
        if layer in LAYERS:
            for fn in REGISTRY[layer]:
                fm, body = fn(name, fm, body)
    out = join_fm(fm, body)
    if out != orig:
        p.write_text(out); changed += 1
        print(f"  edited  {name}")
    else:
        print(f"  same    {name}")
print(f"\n{changed}/20 cards changed by layers {sorted(LAYERS)}")
