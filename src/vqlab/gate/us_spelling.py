#!/usr/bin/env python3
"""vqlab spelling: fail a release on British spellings; --fix rewrites them to US.

Released text (the paper, model cards) is US English. A stray "labelled" or
"neighbouring" is not worth a new paper revision, so it is caught before
release instead (2026-09-28).

    vqlab spelling <file> [<file> ...]          exit 1 and list every hit
    vqlab spelling <file> ... --fix             rewrite in place, then re-check

Two layers:
  * WORDS: an explicit British -> US map (whole words, case preserved).
  * SUFFIX RULES: -isation/-ise/-ised/-ising/-yse... -> -iz/-yz, applied only
    to stems NOT in the allowlist (precise, noise, otherwise, exercise, ...
    are English in both spellings and never touched).
Code spans (`...`) and fenced blocks are skipped, as are URLs, so identifiers
and paths keep their spelling. When a word is correct as written and the
rules still flag it, add it to ALLOW rather than special-casing the caller.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

WORDS = {
    # -our -> -or
    "behaviour": "behavior", "colour": "color", "favour": "favor", "favourable": "favorable",
    "flavour": "flavor", "harbour": "harbor", "honour": "honor", "humour": "humor",
    "labour": "labor", "neighbour": "neighbor", "neighbouring": "neighboring",
    "neighbourhood": "neighborhood", "rumour": "rumor", "savour": "savor", "vapour": "vapor",
    "vigour": "vigor", "endeavour": "endeavor", "armour": "armor", "parlour": "parlor",
    "odour": "odor", "tumour": "tumor",
    # -re -> -er
    "centre": "center", "centred": "centered", "centring": "centering", "fibre": "fiber",
    "litre": "liter", "metre": "meter", "theatre": "theater", "calibre": "caliber",
    "spectre": "specter", "sombre": "somber",
    # doubled l
    "labelled": "labeled", "labelling": "labeling", "modelled": "modeled",
    "modelling": "modeling", "travelled": "traveled", "travelling": "traveling",
    "cancelled": "canceled", "cancelling": "canceling", "signalled": "signaled",
    "signalling": "signaling", "totalled": "totaled", "totalling": "totaling",
    "tunnelled": "tunneled", "tunnelling": "tunneling", "levelled": "leveled",
    "levelling": "leveling", "channelled": "channeled", "fuelled": "fueled",
    "marvellous": "marvelous", "counsellor": "counselor", "jewellery": "jewelry",
    # single l in US
    "enrolment": "enrollment", "fulfil": "fulfill", "fulfilment": "fulfillment",
    "instalment": "installment", "skilful": "skillful", "wilful": "willful",
    # -ence/-ense
    "defence": "defense", "offence": "offense", "licence": "license", "pretence": "pretense",
    # -ogue
    "analogue": "analog", "catalogue": "catalog", "dialogue": "dialog",
    # misc
    "grey": "gray", "programme": "program", "programmes": "programs", "aluminium": "aluminum",
    "ageing": "aging", "judgement": "judgment", "acknowledgement": "acknowledgment",
    "whilst": "while", "amongst": "among", "towards": "toward", "afterwards": "afterward",
    "learnt": "learned", "spelt": "spelled", "burnt": "burned", "dreamt": "dreamed",
    "sceptical": "skeptical", "manoeuvre": "maneuver", "plough": "plow", "tyre": "tire",
    "cheque": "check", "draught": "draft", "storey": "story", "artefact": "artifact",
    "artefacts": "artifacts", "practise": "practice", "practised": "practiced",
    "enquiry": "inquiry", "mould": "mold", "orientated": "oriented",
}
# -ise family: stems whose -ise is correct in US English too.
ALLOW = {
    "advise", "arise", "chastise", "circumcise", "comprise", "compromise", "concise",
    "despise", "devise", "disguise", "enterprise", "excise", "exercise", "expertise",
    "franchise", "improvise", "incise", "merchandise", "noise", "otherwise", "paradise",
    "praise", "precise", "premise", "promise", "raise", "reprise", "revise", "rise",
    "supervise", "surmise", "surprise", "televise", "treatise", "wise", "likewise",
    "clockwise", "stepwise", "pairwise", "elementwise", "layerwise", "rowwise",
    "blockwise", "channelwise", "piecewise", "groupwise", "tensorwise", "bitwise",
    "pointwise", "coordinatewise", "sunrise", "uprise", "cruise", "bruise", "poise",
    "anise", "valise", "demise", "guise", "mise", "sise", "vise", "baise", "noisy",
    "unwise", "otherwise", "promised", "raised", "praised", "revised", "devised",
    "supervised", "unsupervised", "exercised", "surprised", "comprised", "advised",
    "compromised", "disguised", "improvised", "premised", "risen", "arisen",
    "precisely", "concisely", "noises", "denoise", "denoised", "denoising",
    "analyses", "paralyses", "catalyses", "dialyses",  # also the US plural noun
    "trellis", "trellises", "chassis", "premises", "crises", "bases",
}
SUFFIX = [  # (british suffix, us suffix), longest first
    ("isations", "izations"), ("isation", "ization"), ("ising", "izing"), ("ised", "ized"),
    ("ises", "izes"), ("iser", "izer"), ("isers", "izers"), ("ise", "ize"),
    ("ysing", "yzing"), ("ysed", "yzed"), ("yses", "yzes"), ("yse", "yze"),
]
YSE_STEMS = ("analy", "paraly", "cataly", "hydroly", "electroly", "dialy")
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
SKIP_RE = re.compile(r"```.*?```|`[^`\n]*`|https?://\S+|\]\([^)]*\)", re.S)


def _case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[0].isupper():
        return dst[0].upper() + dst[1:]
    return dst


def us_form(word: str) -> str | None:
    """The US spelling of `word`, or None if it is already US (or unknown)."""
    w = word.lower()
    if w in WORDS and WORDS[w] != w:
        return _case(word, WORDS[w])
    if w in ALLOW:
        return None
    for br, us in SUFFIX:
        if w.endswith(br) and len(w) > len(br) + 2:
            stem = w[: -len(br)]
            if br.startswith("y") and not (stem + "y").endswith(YSE_STEMS):
                return None
            base = stem + br
            if base in ALLOW or (stem + "ise") in ALLOW or (stem + "e") in ALLOW:
                return None
            return _case(word, stem + us)
    return None


def scan(text: str):
    """[(offset, word, us)] for every British spelling outside code/URLs."""
    masked = SKIP_RE.sub(lambda m: " " * len(m.group(0)), text)
    hits = []
    for m in WORD_RE.finditer(masked):
        # Judge each part of a hyphenated compound on its own ("layer-wise"
        # is layer + wise; "well-labelled" is well + labelled).
        off = m.start()
        for part in re.split(r"([-'])", m.group(0)):
            us = us_form(part) if part not in "-'" else None
            if us:
                hits.append((off, part, us))
            off += len(part)
    return hits


def fix(text: str) -> str:
    out, last = [], 0
    for off, word, us in scan(text):
        out.append(text[last:off]); out.append(us); last = off + len(word)
    out.append(text[last:])
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab spelling", description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fix", action="store_true", help="rewrite British spellings in place")
    a = ap.parse_args(argv)
    bad = 0
    for f in a.files:
        p = pathlib.Path(f)
        text = p.read_text()
        if a.fix:
            new = fix(text)
            if new != text:
                p.write_text(new)
                print(f"fixed {len(scan(text))} spelling(s) in {p}")
            text = new
        for off, word, us in scan(text):
            line = text.count("\n", 0, off) + 1
            print(f"{p}:{line}: {word} -> {us}")
            bad += 1
    print(f"spelling: {bad} British spelling(s)" if bad else "spelling: US English, PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
