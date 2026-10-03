#!/usr/bin/env python3
"""card: a model card from its sources, not typed by hand.

    vqlab card --artifact DIR --kl KL.json [--kl more.json] --this RUNG
               [--base-model REPO] [--runtime knurlogic|mlx-lm]
               [--label rung="row label" ...] [--extra name=prose,code,lit[,gib]]
               [--extra-md FILE] [--license ID] [--draft] [--out README.md]

Fills the house template (front matter, headline sizes, measured results,
geometry, run it, methodology, paper, support, provenance). Every number
has one source:

  sizes            core.artifact.sizes()  -- the `vqlab size` numbers
  KL table, paired card_tables.render()   -- the `vqlab card-tables` output
  "Measured with"  card_tables.measured_with() over the KL JSON's stamps
  geometry         config.json vq_modules (+ quantization for the skeleton)
  recipe/lineage   vqlab_provenance.json (+ its history file)

Prose a human must write (intro, changelog, notes, limitations, ...) comes
from --extra-md: `## Name` sections, matched to the template's slots by
name (intro, changelog, notes, hardware, speed, verification,
limitations); text before the first heading is the intro, and unmatched
sections go in before Paper. A slot with no prose becomes a marked TODO
block, and the command exits 1 while any TODO remains unless --draft.

Refuses (exit 2) when the KL JSON stamped a fingerprint for --this that is
not --artifact's shard fingerprint, when the rungs were measured under
mixed numerics builds or scorer variants, or when the scorer recorded an
artifact changing during the run. British spellings fail like TODOs.

Prints to stdout. --out writes only the path it names.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from vqlab.core.artifact import sizes
from vqlab.records import card_tables, provenance

GIB = 2 ** 30
TODO = "TODO(card)"
SLOTS = ("intro", "changelog", "notes", "hardware", "speed", "verification", "limitations")
REQUIRED = ("intro", "changelog", "limitations")      # TODO when absent
SPONSOR = """## Support this work

VQLab and these artifacts are built and released independently — the fits,
the measurement harness and the published rungs are one person's compute and
time. If they are useful to you:

- **Sponsor:** [github.com/sponsors/noahzelezny](https://github.com/sponsors/noahzelezny)
- **Contract work:** available for quantization and on-device inference work
  on Apple silicon — custom rungs, per-layer allocation for your model, or
  getting a checkpoint to run well on a Mac.
  Contact: [hello@thedrainflorist.com](mailto:hello@thedrainflorist.com)"""
PAPER = """## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)"""
METHOD = """**Vector quantization instead of scalar rounding.** Each small group of
weights stores one index into a learned codebook of joint patterns, so the
codebook's entries sit where the weight distribution actually is.

**Codebooks are fit in pure weight space**: k-means over the weight
subvectors, no Hessian, no activation statistics, no calibration corpus.

**Sub-byte bit-packing.** Codes are packed to their true bit width; packing
is a pure representation change.

**How it was evaluated.** KL to the teacher on cached top-k logits, on three
house corpora (prose, code, literary), with paired comparisons on identical
positions. KL is the ranking instrument; perplexity is not gated."""


class Refuse(SystemExit):
    def __init__(self, msg):
        print(f"vqlab card: REFUSED: {msg}", file=sys.stderr)
        super().__init__(2)


def todo(what: str) -> str:
    return f"<!-- {TODO} -->\n> **TODO:** {what}\n<!-- /{TODO} -->"


# ------------------------------------------------------------------ sources
def check_kl(kl_paths, this: str, artifact: pathlib.Path) -> dict:
    """Refusals, and the KL JSON facts the card states. Returns
    {"fingerprint": stamped-or-None, "tokens": {corpus: n}, "variants": set}."""
    have = provenance.shard_fingerprint(artifact)
    stamped, found, keys, tokens = set(), False, set(), {}
    for p in kl_paths:
        d = json.load(open(p))
        if d.get("changed_during_run"):
            raise Refuse(f"{p}: the scorer recorded artifacts changing during the run: "
                         f"{d['changed_during_run']}")
        for rung, per in d["table"].items():
            for c, row in per.items():
                keys.add((card_tables.measured_with([row.get("numerics")]),
                          row.get("scorer_variant")))
                if rung == this:
                    fp = (row.get("measured") or {}).get("fingerprint")
                    if fp:
                        stamped.add(fp)
                    tokens[c] = row.get("kl_positions") or row.get("tokens")
        if this in d["table"]:
            found = True
            fp = ((d.get("measured") or {}).get(this) or {}).get("fingerprint")
            if fp:
                stamped.add(fp)
    if not found:
        raise Refuse(f"rung {this!r} is in none of the KL files")
    if len(keys) > 1:
        raise Refuse("KL measured under MIXED builds (not one harness): "
                     + " | ".join(f"{b} variant={v}" for b, v in sorted(keys, key=str)))
    bad = sorted(s for s in stamped if s != have)
    added = None
    if bad and len(stamped) == 1:
        added = text_match(artifact, bad[0])
    if bad and added is None:
        raise Refuse(f"KL for {this!r} was measured on fingerprint {', '.join(bad)}, "
                     f"but {artifact} is {have}: these numbers are not this artifact's")
    return {"fingerprint": next(iter(stamped)) if stamped else None, "tokens": tokens,
            "variant": next(iter(keys))[1], "added_since": added or []}


def _fp(files) -> str:
    import hashlib
    import os
    return hashlib.sha256(json.dumps(
        {f.name: [os.path.getsize(os.path.realpath(f)), provenance._sha(os.path.realpath(f),
                                                                       provenance.HEAD)]
         for f in files}, sort_keys=True).encode()).hexdigest()[:16]


def text_match(artifact: pathlib.Path, want: str):
    """The scored fingerprint can predate a sidecar that holds NO text
    weights (a grafted vision tower, an MTP head). If dropping some set of
    such files reproduces `want`, the text bytes are the scored ones:
    return those file names. Else None. A file holding any text tensor is
    never dropped, so a changed weight still refuses."""
    from itertools import combinations
    from vqlab.core.artifact import read_header, tensor_class
    files = sorted(artifact.glob("*.safetensors"))
    try:
        listed = set(json.load(open(artifact / "model.safetensors.index.json"))["weight_map"].values())
    except (OSError, KeyError, ValueError):
        listed = set()
    # an MTP sidecar (mtp-head*.safetensors outside the index, as `size`
    # counts it) stores its keys without the mtp. prefix (block.*, fc.*), so
    # it is recognised by that rule, not by tensor_class
    side = [f for f in files
            if (f.name.startswith("mtp-head") and f.name not in listed)
            or all(tensor_class(k) != "text" for k in read_header(f))]
    for r in range(1, len(side) + 1):
        for drop in combinations(side, r):
            if _fp([f for f in files if f not in drop]) == want:
                return [f.name for f in drop]
    return None


def _ranges(xs) -> str:
    xs, out, i = sorted(xs), [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)


def geometry(cfg: dict) -> list:
    """[(label, layers, n_modules, experts)] from vq_modules, grouped by
    (dim, K, code bits), most layers first."""
    groups = {}
    # MoE builds keep their codebooks in vq_modules, dense builds in vq_linear
    for name, m in {**(cfg.get("vq_modules") or {}), **(cfg.get("vq_linear") or {})}.items():
        dim, k = m.get("dim"), m.get("k")
        bits = m.get("pack_bits") or (k - 1).bit_length()
        g = groups.setdefault((dim, k, bits), {"layers": set(), "n": 0, "experts": set()})
        hit = re.search(r"layers\.(\d+)\.", name)
        if hit:
            g["layers"].add(int(hit.group(1)))
        g["n"] += 1
        if m.get("experts"):
            g["experts"].add(m["experts"])
    return [(f"d{d}/K{k} ({b}-bit codes, {b / d:g} code bits per weight)",
             g["layers"], g["n"], g["experts"])
            for (d, k, b), g in sorted(groups.items(), key=lambda kv: -len(kv[1]["layers"]))]


def skeleton(cfg: dict) -> str | None:
    q = cfg.get("quantization") or {}
    if not q:
        return None
    mode = q.get("mode", "affine")
    per = sum(isinstance(v, dict) for v in q.values())
    return (f"{q.get('bits')}-bit {mode} (group {q.get('group_size')})"
            + (f", with {per} per-module overrides in `config.json`" if per else ""))


def lineage(art: pathlib.Path):
    """(records oldest first, recipe record or None) from the build record
    and its history file."""
    recs = []
    h = art / provenance.HISTORY
    if h.exists():
        recs += [json.loads(x) for x in h.read_text().splitlines() if x.strip()]
    if (art / provenance.RECORD).exists():
        recs.append(provenance.load(art))
    recipe = next((r for r in recs if r["tool"]["name"].split()[0] in
                   ("fit-moe", "geo-build", "mix", "build-dense-vq", "fit-dense")), None)
    return recs, recipe


def read_extra(path) -> dict:
    out, cur = {}, "intro"
    if not path:
        return out
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
            continue
        out.setdefault(cur, []).append(line)
    return {k: "\n".join(v).strip() for k, v in out.items() if "\n".join(v).strip()}


def _license(art: pathlib.Path, given):
    if given:
        return given
    lic = art / "LICENSE"
    if lic.exists():
        head = lic.read_text()[:400].lower()
        for key, ident in (("mit license", "mit"), ("apache license", "apache-2.0")):
            if key in head:
                return ident
    return None


# ------------------------------------------------------------------- render
def build(a) -> str:
    art = pathlib.Path(a.artifact)
    cfg = json.loads((art / "config.json").read_text())
    kl = check_kl(a.kl, a.this, art)
    s = sizes(art)
    extra = read_extra(a.extra_md)
    slot = {k.lower(): v for k, v in extra.items() if k.lower() in SLOTS}
    loose = {k: v for k, v in extra.items() if k.lower() not in SLOTS}
    recs, recipe = lineage(art)

    owner, _, title = art.name.partition("--")
    if not title:
        owner, title = "TheDrainFlorist", art.name
    repo = f"{owner}/{title}"
    base = a.base_model or next((i.get("hf_repo") for r in recs for i in r.get("inputs", [])
                                 if i.get("hf_repo")), None)
    lic = _license(art, a.license)
    mt = cfg.get("model_type")

    def sect(name, what):
        return slot.get(name) or (todo(what) if name in REQUIRED else None)

    L = ["---", "language:", "- en", f"license: {lic or TODO}", "library_name: mlx",
         "pipeline_tag: text-generation"]
    if base:
        L += [f"base_model: {base}", "base_model_relation: quantized"]
    L += ["tags:", "- mlx", "- quantized", "- vector-quantization", "- apple-silicon"]
    if mt:
        L.append(f"- {mt}")
    L += ["---", "", f"# {title}", ""]

    mtp = s["mtp"] + s["mtp_sidecar"]
    L.append(f"**{s['text'] / GIB:.1f} GiB text weights.**")
    parts = []
    parts.append(f"vision tower +{s['tower'] / GIB:.2f} GiB" if s["tower"] else "no vision tower")
    if mtp:
        parts.append(f"MTP draft head +{mtp / GIB:.2f} GiB")
    parts.append(f"full download {s['download'] / GIB:.1f} GiB")
    line = "; ".join(parts)
    L += [line[0].upper() + line[1:] + ".", ""]
    L += [sect("intro", "intro: what this build is, of what, and the one-line claim the "
                        "table supports. Facts only."), ""]
    L += ["## Changelog", "", sect("changelog", "changelog: dated entry for this release."), ""]

    L += ["## Measured results", ""]
    toks = sorted({t for t in kl["tokens"].values() if t})
    L += [f"KL to the teacher in millinats per token (lower is closer), "
          f"{'/'.join(str(t) for t in toks) or '?'} tokens per corpus, every row on the same "
          f"cache and positions. KL is the gate; perplexity is not."
          + (f" Scorer variant: `{kl['variant']}`." if kl["variant"] else ""), ""]
    labels = dict(x.split("=", 1) for x in a.label)
    L += [card_tables.render(a.kl, labels, a.this, a.extra), ""]
    if kl["fingerprint"]:
        L += [f"The **{labels.get(a.this, a.this)}** row was measured on this artifact's "
              f"exact text weight bytes (shard fingerprint `{kl['fingerprint']}`)"
              + (f"; added since, holding no text weights: "
                 f"{', '.join(f'`{n}`' for n in kl['added_since'])}" if kl["added_since"] else "")
              + ".", ""]
    if slot.get("notes"):
        L += [slot["notes"], ""]
    for name in ("hardware", "speed"):
        if slot.get(name):
            L += [f"## {name.capitalize()}", "", slot[name], ""]

    L += ["## Run it", "", "Requires a Mac with Apple Silicon (macOS, MLX).", ""]
    if a.runtime == "knurlogic":
        L += ["Runs on [Knurlogic](https://github.com/noahzelezny/Knurlogic), which runs the "
              "`model.py` this repo ships (declared by `model_file` in `config.json`).", "",
              "```bash", "pip install knurlogic",
              f"hf download {repo} --local-dir ~/Knurlogic/Models/{title}",
              f"knurlogic serve ~/Knurlogic/Models/{title}", "```", "",
              "Chat at http://127.0.0.1:8080/, or point any OpenAI-, Anthropic- or "
              "Ollama-compatible client at it (model `local`).", ""]
    else:
        L += ["Stock `mlx-lm`; the VQ runtime ships inside the checkpoint as `model.py`.", "",
              "```bash", "pip install mlx-lm",
              f"mlx_lm.generate --model {repo} --trust-remote-code --prompt \"Hello\"",
              "```", ""]

    L += ["## Methodology", ""]
    geo = geometry(cfg)
    if geo:
        nlay = cfg.get("num_hidden_layers")
        key = "vq_modules" if cfg.get("vq_modules") else "vq_linear"
        nlay = nlay or (cfg.get("text_config") or {}).get("num_hidden_layers")
        L += [f"Vector-quantized modules ({sum(g[2] for g in geo)} in `{key}`"
              + (f", over {nlay} layers" if nlay else "") + "):", ""]
        moe = any(set(ex) - {1} for *_, ex in geo)       # dense: every module is one "expert"
        L += (["| geometry | layers | modules | experts per module |", "|---|---|---|---|"] if moe
              else ["| geometry | layers | modules |", "|---|---|---|"])
        for label, lay, n, ex in geo:
            L.append(f"| {label} | {_ranges(lay) or '-'} | {n} |"
                     + (f" {', '.join(map(str, sorted(ex))) or '-'} |" if moe else ""))
        L.append("")
    sk = skeleton(cfg)
    if sk:
        L += [f"Everything else (the skeleton): {sk}.", ""]
    L += [METHOD, ""]
    if slot.get("verification"):
        L += ["## Verification", "", slot["verification"], ""]
    L += ["## Limitations", "", sect("limitations", "limitations: runtime needs, what was "
                                     "not measured (task benchmarks, absolute speed)."), ""]
    for k, v in loose.items():
        L += [f"## {k}", "", v, ""]
    L += [PAPER, "", SPONSOR, "", "## Provenance", ""]
    if base:
        L.append(f"Base model: [{base}](https://huggingface.co/{base})"
                 + (f" — **{lic.upper() if lic == 'mit' else lic}**." if lic else ".")
                 + " This is a quantized derivative and inherits that license.")
    else:
        L.append(todo("base model (--base-model REPO)"))
    if recipe:
        argv = " ".join(recipe["tool"]["argv"][:1] + [x for x in recipe["tool"]["argv"][1:]
                                                     if not x.startswith("/")])
        L.append(f"Recipe: `{argv}` (VQLab {recipe['code'].get('commit', '?')[:10]}, "
                 f"{recipe['created'][:10]}).")
    if recs:
        L += [f"Build record: `vqlab_provenance.json` (id `{recs[-1]['id'][:12]}`); lineage: "
              + " -> ".join(f"{r['tool']['name']} ({r['created'][:10]})" for r in recs) + "."]
    L += ["", "Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab)."]
    return "\n".join(L).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab card", description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--kl", action="append", required=True)
    ap.add_argument("--this", required=True, help="the KL JSON's rung that IS --artifact")
    ap.add_argument("--base-model")
    ap.add_argument("--runtime", choices=("knurlogic", "mlx-lm"), default="knurlogic")
    ap.add_argument("--label", action="append", default=[], help='rung="row label"')
    ap.add_argument("--extra", action="append", default=[],
                    help="name=prose,code,lit[,gib]: a comparator row measured elsewhere")
    ap.add_argument("--extra-md", help="human prose: `## Name` sections for the template slots")
    ap.add_argument("--license", help="license id (default: read from the artifact's LICENSE)")
    ap.add_argument("--draft", action="store_true", help="exit 0 even with TODOs / spellings")
    ap.add_argument("--out", help="write here instead of stdout (only this path is written)")
    a = ap.parse_args(argv)
    text = build(a)
    from vqlab.gate import us_spelling
    spell = us_spelling.scan(text)
    for off, word, us in spell:
        print(f"card:{text.count(chr(10), 0, off) + 1}: {word} -> {us}", file=sys.stderr)
    n_todo = text.count(f"<!-- {TODO} -->") + text.count(f"license: {TODO}")
    if a.out:
        pathlib.Path(a.out).write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    if n_todo or spell:
        print(f"vqlab card: {n_todo} TODO block(s), {len(spell)} British spelling(s)"
              + (" (draft)" if a.draft else "; not releasable"), file=sys.stderr)
        return 0 if a.draft else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
