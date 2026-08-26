#!/usr/bin/env python3
"""Build a PUBLIC code-scoring corpus, with a provenance manifest.

The paper's code-perplexity numbers were measured on a private code corpus
that is not redistributable, so it does not ship (the prose and literary
corpora do, and the prose column is fully reproducible). This script builds
a public replacement from an Apache-2.0 project checkout, with per-file
sha256, byte counts and attribution recorded in a manifest.

**A rebuilt corpus is a DIFFERENT INSTRUMENT.** Scores measured on it are
internally comparable (model vs model on this corpus) but are NOT comparable
to the paper's code-ppl column, which used the private corpus. Do not put
numbers from the two corpora in the same table.

    python scripts/make_code_corpus.py --repo <checkout-dir> \\
        --files generate.py task_group.py ... --out my_code_corpus.txt

The manifest (written next to the corpus) is the frozen definition of the
instrument: if two people's manifests differ, they hold different
instruments, whatever the filenames say.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True,
                    help="checkout of the source project (record its commit "
                         "in --attribution; this script records file hashes)")
    ap.add_argument("--files", nargs="+", required=True,
                    help="file names to include, located by exact basename "
                         "search under --repo; each must match exactly one "
                         "file or be given as a repo-relative path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attribution", required=True,
                    help='e.g. "exo (github.com/exo-explore/exo) @ <commit>, '
                         'Apache-2.0" — copied verbatim into the manifest '
                         "and the corpus header. For Apache-2.0 sources, "
                         "keep the project's LICENSE/NOTICE alongside any "
                         "redistribution of the built corpus.")
    a = ap.parse_args()
    repo = pathlib.Path(a.repo)
    out = pathlib.Path(a.out)

    chunks, manifest = [], []
    for spec in a.files:
        p = repo / spec
        if not p.exists():
            hits = [h for h in repo.rglob(spec) if h.is_file()]
            if len(hits) != 1:
                sys.exit(f"FAIL: {spec!r} matched {len(hits)} files under "
                         f"{repo} — give a repo-relative path instead.")
            p = hits[0]
        text = p.read_text(errors="replace")
        lang = {"py": "python", "rs": "rust", "js": "js"}.get(
            p.suffix.lstrip("."), p.suffix.lstrip("."))
        chunks.append(f"// ==== {lang}: {p.name} ====\n\n{text}\n")
        manifest.append({
            "file": str(p.relative_to(repo)),
            "bytes": len(text.encode()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        })

    header = (f"// PUBLIC CODE CORPUS — built by make_code_corpus.py\n"
              f"// source: {a.attribution}\n"
              f"// This is a DIFFERENT instrument from the paper's private "
              f"code corpus;\n// scores are not comparable across the two.\n\n")
    out.write_text(header + "".join(chunks))
    man = {"attribution": a.attribution, "files": manifest,
           "total_bytes": sum(m["bytes"] for m in manifest),
           "note": "different instrument from the paper's private code corpus"}
    mp = out.with_suffix(out.suffix + ".manifest.json")
    mp.write_text(json.dumps(man, indent=1))
    print(f"wrote {out} ({man['total_bytes']} bytes, {len(manifest)} files)")
    print(f"wrote {mp}")
    print("Reminder: scores on this corpus do NOT compare to the paper's "
          "code column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
