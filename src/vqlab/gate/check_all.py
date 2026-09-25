#!/usr/bin/env python3
"""vqlab check — run the release gates that need no source model.

Composite of `check-release` (every file a downloader needs exists and
functions) and `check-bundle` (the shipped runtime matches this repo's).
Runs both, reports both, and fails if either fails — never stops at the
first failure, because knowing only that something broke is less useful
than knowing everything that broke.

Since 2026-09-19 it also runs the VISION SURFACE check on any artifact whose
config declares a vision or audio tower. That gate is here, in the cheap
no-weights composite, because of F153: 17 of 20 shipped multimodal bundles
carried a vision tower their runtime could not reach, and EVERY existing gate
passed them. They passed honestly — check-release found every file present,
check-bundle found the right runtime text, and `smoke` generated a token —
because a text-only arch serves text perfectly well. Nothing in the set ever
asked whether the shipped vision weights were reachable, so for three weeks
nothing said otherwise.

The check is the surface arm of `vision-smoke` (`--static`): import the
bundle's own model.py and demand the multimodal surface. It needs no weights
and costs about a second, which is why it can sit in the gate everyone runs
rather than in a step reserved for a big box. Text-only artifacts skip it by
their own config, so this costs them nothing.

The rule it encodes: an artifact that SHIPS vision weights must prove its
runtime can reach them. Shipping 1,400 tensors the loader discards is not a
configuration choice, it is a defect, and it should never again be possible
to publish one with a green board.

NOT included, deliberately, because each needs an input this command does
not have:
  - the outlier gate       (`vqlab verify`, needs the bf16 source)
  - generation             (`vqlab smoke`, needs the model resident)
  - the vision LOAD+GRAPH arms (`vqlab vision-smoke` without `--static`,
    needs the model resident — the surface arm here is necessary, not
    sufficient: it proves the runtime HAS the tower, not that an image
    reaches the graph)
  - bundled-kernel accept  (`vqlab bundle-accept`)
  - comparator parity      (`vqlab check-comparator`, needs the teacher)
A pass here means the artifact is well-formed and ships the right runtime.
It does NOT mean the artifact is correct or that it can serve.

    vqlab check <artifact>
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab._layout import find as _find  # noqa: E402
# (gate name, script, how it takes the artifact). vision-smoke predates this
# composite and takes a POSITIONAL artifact; the others take --artifact.
# Encoded per-gate rather than normalised, so adding a gate never silently
# passes the path in a form the script ignores.
GATES = [
    ("check-release", "check_release.py", "flag", []),
    ("check-bundle", "check_bundle.py", "flag", []),
    # Surface arm only: no weights, ~1s, self-skips on text-only artifacts.
    ("vision-surface", "vision_smoke.py", "positional", ["--static"]),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab check",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    a = ap.parse_args(argv)

    results = []
    for name, script, style, extra in GATES:
        print(f"\n=== {name} ===", flush=True)
        if style == "positional":
            argv = [a.artifact, *extra]
        else:
            argv = ["--artifact", a.artifact, *extra]
        p = subprocess.run([sys.executable, str(_find(script)), *argv])
        results.append((name, p.returncode == 0))

    print("\n=== summary ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} gate(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll release gates passed. Still required before release: "
          "`vqlab verify` (outlier gate, on a box that did not fit it), "
          "`vqlab smoke` (one token through the shipping runtime), and for a "
          "multimodal artifact `vqlab vision-smoke` WITHOUT --static — the "
          "surface arm above proves the runtime has a tower, not that an "
          "image moves the logits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
