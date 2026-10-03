#!/usr/bin/env python3
"""card-tables: the measured tables of a model card, generated, not typed.

    vqlab card-tables --kl kl_ladder.json [--kl more.json ...]
                      [--label rung="row label" ...] [--this rung] [--extra name=prose,code,lit]

Prints Markdown: the KL table (one row per rung across every --kl file, with
text GiB measured from each rung's own headers, prose / code / literary and
the mean), and the paired deltas each file recorded against its reference.
--this bolds the released rung. --extra adds a row measured elsewhere (a
comparator), named and passed as numbers, so it is visibly not from these
files.

Two card near-misses on 2026-10-02 came from typing these by hand: a wrong
acknowledgment carried between cards, and a 'text weights' figure misread
because the 397B indexes its MTP head. Numbers here come only from the JSON
the scorer wrote and the artifact's own headers.
"""
from __future__ import annotations

import argparse
import json
import sys

from vqlab.core.artifact import sizes

CORPORA = ("prose", "code", "lit")
GIB = 2 ** 30


def _rows(paths):
    rows, paired, builds = {}, [], []
    for p in paths:
        d = json.load(open(p))
        for rung, per in d["table"].items():
            vals = [per[c]["mean_kl_millinats"] for c in CORPORA if c in per]
            if len(vals) != len(CORPORA):
                raise SystemExit(f"{p}: rung {rung} lacks a corpus ({sorted(per)})")
            rows[rung] = {"vals": vals, "path": d["rungs"][rung]}
            builds += [per[c].get("numerics") for c in CORPORA]
        for rung, per in (d.get("paired") or {}).items():
            paired.append((rung, d["reference"], per))
    return rows, paired, builds


def measured_with(builds) -> str:
    """One line naming the numerics build behind the numbers (F194: a KL
    number is only meaningful for the build that produced it). Every
    distinct build is named; a mixed table says so."""
    from vqlab.core.numerics import describe
    seen = []
    for b in builds:
        line = describe(b)
        if line not in seen:
            seen.append(line)
    if len(seen) == 1:
        return f"Measured with: {seen[0]}."
    return "Measured with MIXED builds (not one harness): " + " | ".join(seen) + "."


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab card-tables", description=__doc__.split("\n")[0])
    ap.add_argument("--kl", action="append", required=True)
    ap.add_argument("--label", action="append", default=[], help='rung="row label"')
    ap.add_argument("--this", help="the rung being released (bolded)")
    ap.add_argument("--extra", action="append", default=[],
                    help="name=prose,code,lit[,gib] for a row measured elsewhere")
    a = ap.parse_args(argv)
    labels = dict(x.split("=", 1) for x in a.label)
    print(render(a.kl, labels, a.this, a.extra))
    return 0


def render(kl_paths, labels=None, this=None, extras=()) -> str:
    """The card's tables as one Markdown string (`vqlab card` embeds this)."""
    labels = labels or {}
    rows, paired, builds = _rows(kl_paths)
    stamp = measured_with(builds)
    out = ["| build | GiB (text) | prose | code | literary | mean |", "|---|---|---|---|---|---|"]
    for x in extras:
        name, nums = x.split("=", 1)
        v = [float(n) for n in nums.split(",")]
        gib = f"{v[3]:.1f}" if len(v) > 3 else "?"
        out.append(f"| {name} | {gib} | " + " | ".join(f"{n:.1f}" for n in v[:3])
                   + f" | {sum(v[:3]) / 3:.1f} |")
    for rung, r in sorted(rows.items(), key=lambda kv: -sum(kv[1]["vals"])):
        try:
            gib = f"{sizes(r['path'])['text'] / GIB:.1f}"
        except (SystemExit, OSError):
            gib = "?"
        cells = [gib] + [f"{v:.1f}" for v in r["vals"]] + [f"{sum(r['vals']) / 3:.1f}"]
        name = labels.get(rung, rung)
        if rung == this:
            name, cells = f"**{name}**", [f"**{c}**" for c in cells]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", stamp]
    if paired:
        out += ["", "Paired deltas on identical positions (this rung minus reference; "
                "|t|>2 is a difference):", "",
                "| rung | vs | prose | code | literary |", "|---|---|---|---|---|"]
        for rung, ref, per in paired:
            cells = [f"{per[c]['delta']:+.1f} (t {per[c]['t']:+.1f})" for c in CORPORA]
            out.append(f"| {labels.get(rung, rung)} | {labels.get(ref, ref)} | "
                       + " | ".join(cells) + " |")
        out += ["", stamp]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
